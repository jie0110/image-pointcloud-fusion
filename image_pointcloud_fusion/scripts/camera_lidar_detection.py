#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import os
import torch
from ultralytics import YOLO

class YoloLidarFusion:
    def __init__(self):
        rospy.init_node('yolo_lidar_fusion', anonymous=True)
        
        # 获取参数
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/color/image_raw')
        self.lidar_topic = rospy.get_param('~lidar_topic', '/livox/lidar')
        self.detection_image_topic = rospy.get_param('~detection_image_topic', '/detection/image')
        self.bbox_markers_topic = rospy.get_param('~bbox_markers_topic', '/detection/bbox_markers')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'livox_frame')
        self.detection_threshold = rospy.get_param('~detection_threshold', 0.7)
        self.yolo_model_path = rospy.get_param('~yolo_model_path', 'yolo11s.pt')
        
        # 相机内参矩阵
        fx = rospy.get_param('~fx', 909.783)
        fy = rospy.get_param('~fy', 909.004)
        cx = rospy.get_param('~cx', 650.365)
        cy = rospy.get_param('~cy', 381.295)
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 激光雷达到相机的外参矩阵
        extrinsic_matrix = rospy.get_param('~extrinsic_matrix', None)
        if extrinsic_matrix is None:
            self.extrinsic_matrix = np.array([
                [0.0, -1.0, 0.0, 0.03],
                [0.0, 0.0, -1.0, -0.022],
                [1.0, 0.0, 0.0, -0.04],
                [0, 0, 0, 1]
            ])
        else:
            self.extrinsic_matrix = np.array(extrinsic_matrix).reshape(4, 4)
        
        # 相机到激光雷达的变换矩阵（用于将相机坐标转换到激光雷达坐标）
        self.camera_to_lidar_matrix = np.linalg.inv(self.extrinsic_matrix)
        
        rospy.loginfo("Camera matrix: \n{}".format(self.camera_matrix))
        rospy.loginfo("Extrinsic matrix: \n{}".format(self.extrinsic_matrix))
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 加载YOLOv11模型
        try:
            rospy.loginfo("Loading YOLOv11 model from: {}".format(self.yolo_model_path))
            self.yolo_model = self.load_yolo_model(self.yolo_model_path)
            rospy.loginfo("YOLOv11 model loaded successfully")
        except Exception as e:
            rospy.logerr("Failed to load YOLOv11 model: {}".format(e))
            raise
        
        # 创建发布者
        self.detection_image_pub = rospy.Publisher(self.detection_image_topic, Image, queue_size=1)
        self.bbox_markers_pub = rospy.Publisher(self.bbox_markers_topic, MarkerArray, queue_size=1)
        
        # 创建订阅者并进行时间同步
        self.image_sub = Subscriber(self.camera_topic, Image)
        self.lidar_sub = Subscriber(self.lidar_topic, PointCloud2)
        
        # 使用Approximate Time Synchronizer同步数据
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("YoloLidarFusion node initialized")
    
    def load_yolo_model(self, model_path):
        """加载YOLO模型(使用ultralytics库)"""
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        # 使用ultralytics库加载YOLO模型
        try:
            # 导入ultralytics
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(model_path)
            
            # 设置推理参数
            model.conf = self.detection_threshold  # 置信度阈值
            rospy.loginfo(f"Successfully loaded YOLO model from: {model_path}")
            rospy.loginfo(f"Model type: {type(model).__name__}")
            return model
            
        except Exception as e:
            rospy.logerr(f"Error loading YOLO model with ultralytics: {e}")
            raise
    
    def callback(self, image_msg, pointcloud_msg):
        """处理同步到的图像和点云数据"""
        try:
            start_time = time.time()
            
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # 从点云消息中提取点云数据
            points_list = []
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append(point)
            
            # 将点云数据转换为NumPy数组
            points = np.array(points_list)
            
            # 使用YOLOv11进行目标检测
            detections = self.detect_objects(cv_image)
            
            # 在图像上绘制检测结果
            detection_image = self.draw_detections_on_image(cv_image, detections)
            
            # 为点云中的检测对象创建3D边界框
            markers = self.create_3d_bboxes(points, detections, pointcloud_msg.header)
            
            # 发布检测结果图像
            detection_image_msg = self.bridge.cv2_to_imgmsg(detection_image, "bgr8")
            detection_image_msg.header = image_msg.header
            self.detection_image_pub.publish(detection_image_msg)
            
            # 发布3D边界框标记
            self.bbox_markers_pub.publish(markers)
            
            processing_time = time.time() - start_time
            rospy.logdebug("Processing time: {:.3f} seconds".format(processing_time))
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        except Exception as e:
            rospy.logerr("Error in callback: {0}".format(e))
    
    def detect_objects(self, image):
        """使用ultralytics YOLO进行目标检测"""
        try:
            # 使用YOLO模型进行推理
            results = self.yolo_model(image)
            
            # 准备检测结果列表
            detections = []
            
            # ultralytics YOLO返回Results对象列表
            for result in results:
                # 获取检测框
                boxes = result.boxes
                
                # 遍历每个检测框
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度
                    conf = box.conf[0].cpu().numpy()
                    
                    # 获取类别ID和名称
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
            
            rospy.loginfo(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            rospy.logerr(f"Error during object detection: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return []
    
    def draw_detections_on_image(self, image, detections):
        """在图像上绘制检测结果"""
        result_image = image.copy()
        
        # 定义类别-颜色映射
        colors = {
            'person': (0, 255, 0),     # 绿色
            'car': (0, 0, 255),        # 红色
            'truck': (255, 0, 0),      # 蓝色
            'bicycle': (255, 255, 0),  # 青色
            'dog': (255, 0, 255),      # 紫色
            'cat': (0, 255, 255),      # 黄色
        }
        
        default_color = (255, 255, 255)  # 白色，用于未定义颜色的类别
        
        for det in detections:
            # 提取边界框坐标
            x1, y1, x2, y2 = det['bbox']
            
            # 获取类别名称和置信度
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 选择颜色
            color = colors.get(class_name.lower(), default_color)
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = "{}: {:.2f}".format(class_name, confidence)
            
            # 计算标签的大小
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 绘制标签背景
            cv2.rectangle(result_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def project_points_to_image(self, points):
        """将点云投影到图像平面上"""
        # 为点云添加第四个坐标（齐次坐标）
        points_homogeneous = np.ones((points.shape[0], 4))
        points_homogeneous[:, :3] = points
        
        # 使用外参矩阵将点从激光雷达坐标系变换到相机坐标系
        points_camera = np.dot(self.extrinsic_matrix, points_homogeneous.T).T
        
        # 过滤掉相机后方的点（Z < 0）
        valid_indices = points_camera[:, 2] > 0
        points_camera = points_camera[valid_indices]
        valid_original_points = points[valid_indices]
        
        if len(points_camera) == 0:
            return None, None, None
        
        # 归一化坐标
        points_normalized = points_camera[:, :3] / points_camera[:, 2:3]
        
        # 使用相机内参矩阵将归一化坐标投影到像素坐标
        pixel_coords = np.dot(self.camera_matrix, points_normalized[:, :3].T).T
        
        # 取整得到像素坐标
        pixel_coords = np.round(pixel_coords[:, :2]).astype(np.int32)
        
        return pixel_coords, valid_original_points, points_camera[:, 2]  # 返回像素坐标、对应的原始点和深度
    
    def create_3d_bboxes(self, points, detections, header):
        """为检测到的对象创建3D边界框"""
        # 将点云投影到图像平面
        pixel_coords, valid_points, depths = self.project_points_to_image(points)
        
        if pixel_coords is None:
            return MarkerArray()
        
        markers_array = MarkerArray()
        marker_id = 0
        
        # 为每个检测结果创建3D边界框
        for det in detections:
            # 获取2D边界框坐标
            x1, y1, x2, y2 = det['bbox']
            
            # 获取类别名称
            class_name = det['class_name']
            
            # 找出落在边界框内的点
            mask = ((pixel_coords[:, 0] >= x1) & (pixel_coords[:, 0] <= x2) & 
                    (pixel_coords[:, 1] >= y1) & (pixel_coords[:, 1] <= y2))
            
            # 如果边界框内没有点，则跳过
            if not np.any(mask):
                continue
            
            # 获取边界框内的点
            bbox_points = valid_points[mask]
            
            # 计算点云的边界框
            if len(bbox_points) < 5:  # 需要足够的点来确定边界
                continue
                
            # 计算3D边界框
            x_min, y_min, z_min = np.min(bbox_points, axis=0)
            x_max, y_max, z_max = np.max(bbox_points, axis=0)
            
            # 创建边界框标记
            marker = Marker()
            marker.header = header
            marker.ns = "detection"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # 设置边界框中心位置
            marker.pose.position.x = (x_min + x_max) / 2
            marker.pose.position.y = (y_min + y_max) / 2
            marker.pose.position.z = (z_min + z_max) / 2
            
            # 设置边界框方向（默认为无旋转）
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 设置边界框尺寸
            marker.scale.x = x_max - x_min
            marker.scale.y = y_max - y_min
            marker.scale.z = z_max - z_min
            
            # 设置边界框颜色（根据类别）
            if class_name.lower() == 'person':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif class_name.lower() == 'car':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif class_name.lower() == 'truck':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            marker.color.a = 0.5  # 半透明
            
            # 设置标记的持续时间
            marker.lifetime = rospy.Duration(0.1)
            
            # 添加标记到数组
            markers_array.markers.append(marker)
            
            # 创建标签标记
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "detection_label"
            text_marker.id = marker_id + 10000  # 避免ID冲突
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # 设置标签位置（在边界框顶部）
            text_marker.pose.position.x = (x_min + x_max) / 2
            text_marker.pose.position.y = (y_min + y_max) / 2
            text_marker.pose.position.z = z_max + 0.2  # 稍微高于边界框
            
            # 设置标签内容
            text_marker.text = "{}: {:.2f}".format(class_name, det['confidence'])
            
            # 设置标签大小
            text_marker.scale.z = 0.5  # 文本高度
            
            # 设置标签颜色（白色）
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # 设置标记的持续时间
            text_marker.lifetime = rospy.Duration(0.1)
            
            # 添加标记到数组
            markers_array.markers.append(text_marker)
            
            marker_id += 1
        
        return markers_array

def main():
    try:
        yolo_lidar_fusion = YoloLidarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()