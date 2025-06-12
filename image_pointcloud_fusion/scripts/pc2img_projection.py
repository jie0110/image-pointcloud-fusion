#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber

class LidarCameraProjection:
    def __init__(self):
        rospy.init_node('lidar_camera_projection', anonymous=True)
        
        # 获取参数
        self.camera_frame = rospy.get_param('~camera_frame', 'camera')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'livox_frame')
        
        # 相机内参矩阵
        self.camera_matrix = np.array([
            [909.783, 0, 650.365],
            [0, 909.004, 381.295],
            [0, 0, 1]
        ])
        
        # 激光雷达到相机的外参矩阵
        self.extrinsic_matrix = np.array([
            [0.0, -1.0, 0.0, 0.03],
            [0.0, 0.0, -1.0, -0.022],
            [1.0, 0.0, 0.0, -0.04],
            [0, 0, 0, 1]
        ])
        
        rospy.loginfo("Camera matrix: \n{}".format(self.camera_matrix))
        rospy.loginfo("Extrinsic matrix: \n{}".format(self.extrinsic_matrix))
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 创建发布者
        self.projected_image_pub = rospy.Publisher("/camera/projected_points_image", Image, queue_size=1)
        self.depth_image_pub = rospy.Publisher("/camera/depth_image", Image, queue_size=1)
        
        # 创建订阅者并进行时间同步
        self.image_sub = Subscriber("/camera/color/image_raw", Image)
        self.lidar_sub = Subscriber("/livox/lidar", PointCloud2)
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("LidarCameraProjection node initialized")
    
    def callback(self, image_msg, pointcloud_msg):
        """处理同步到的图像和点云数据"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # 从点云消息中提取点云数据
            points_list = []
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append(point)
            
            # 将点云数据转换为NumPy数组
            points = np.array(points_list)
            
            # 投影点云到图像并创建深度图
            projected_image, depth_image = self.project_points_to_image(points, cv_image)
            
            # 发布投影后的图像和深度图
            self.publish_images(projected_image, depth_image, image_msg.header)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        except Exception as e:
            rospy.logerr("Error: {0}".format(e))
    
    def project_points_to_image(self, points, image):
        """将点云投影到图像平面上，并创建深度图"""
        h, w, _ = image.shape
        
        # 创建深度图，初始化为无穷大
        depth_image = np.full((h, w), np.inf, dtype=np.float32)
        
        # 创建投影图像的副本
        projected_image = image.copy()
        
        # 为点云添加第四个坐标（齐次坐标）
        points_homogeneous = np.ones((points.shape[0], 4))
        points_homogeneous[:, :3] = points
        
        # 使用外参矩阵将点从激光雷达坐标系变换到相机坐标系
        points_camera = np.dot(self.extrinsic_matrix, points_homogeneous.T).T
        
        # 过滤掉相机后方的点（Z < 0）
        valid_indices = points_camera[:, 2] > 0
        points_camera = points_camera[valid_indices]
        
        if len(points_camera) == 0:
            rospy.logwarn("No points in front of the camera")
            return projected_image, depth_image
        
        # 归一化坐标
        points_normalized = points_camera[:, :3] / points_camera[:, 2:3]
        
        # 使用相机内参矩阵将归一化坐标投影到像素坐标
        pixel_coords = np.dot(self.camera_matrix, points_normalized[:, :3].T).T
        
        # 取整得到像素坐标
        pixel_coords = np.round(pixel_coords[:, :2]).astype(np.int32)
        
        # 提取Z值（深度）
        depths = points_camera[:, 2]
        
        # 创建彩色映射，用于可视化
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        depth_range = max_depth - min_depth
        
        # 在点云范围内标记点，同时更新深度图
        for i, (u, v) in enumerate(pixel_coords):
            # 检查像素坐标是否在图像范围内
            if 0 <= u < w and 0 <= v < h:
                # 如果该像素当前深度值更小，则更新
                current_depth = depths[i]
                if current_depth < depth_image[v, u]:
                    depth_image[v, u] = current_depth
                    
                    # 根据深度值计算颜色（越近越红，越远越蓝）
                    normalized_depth = (current_depth - min_depth) / depth_range if depth_range > 0 else 0
                    color = self.depth_to_color(normalized_depth)
                    
                    # 在图像上绘制点
                    cv2.circle(projected_image, (u, v), 2, color, -1)
        
        # 处理深度图，将无穷大值设为0（或者您想要的其他值）
        depth_image[np.isinf(depth_image)] = 0
        
        # 归一化深度图以便可视化（可选）
        depth_vis = (depth_image - min_depth) / depth_range if depth_range > 0 else depth_image
        depth_vis = (depth_vis * 255).astype(np.uint8)
        # 应用颜色映射
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        return projected_image, depth_image
    
    def depth_to_color(self, normalized_depth):
        """将归一化深度值转换为BGR颜色"""
        # 使用jet颜色映射，近:红色，远:蓝色
        if normalized_depth < 0.25:
            # 红色到黄色
            r = 255
            g = int(normalized_depth * 4 * 255)
            b = 0
        elif normalized_depth < 0.5:
            # 黄色到绿色
            r = int(255 - (normalized_depth - 0.25) * 4 * 255)
            g = 255
            b = 0
        elif normalized_depth < 0.75:
            # 绿色到青色
            r = 0
            g = 255
            b = int((normalized_depth - 0.5) * 4 * 255)
        else:
            # 青色到蓝色
            r = 0
            g = int(255 - (normalized_depth - 0.75) * 4 * 255)
            b = 255
        
        return (b, g, r)  # OpenCV使用BGR格式
    
    def publish_images(self, projected_image, depth_image, header):
        """发布投影图像和深度图"""
        try:
            # 发布投影图像
            projected_msg = self.bridge.cv2_to_imgmsg(projected_image, "bgr8")
            projected_msg.header = header
            self.projected_image_pub.publish(projected_msg)
            
            # 发布深度图
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
            depth_msg.header = header
            self.depth_image_pub.publish(depth_msg)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    try:
        lidar_camera_projection = LidarCameraProjection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()