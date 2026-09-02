#!/opt/yolo_env/bin/python3



import torch
from ultralytics import YOLO

import json
import traceback

import cv2
import numpy as np

import rospy
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError


class CameraToYOLO:
    def __init__(self):
        rospy.init_node('camera_to_yolo', anonymous=False)

        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        #self.depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')

        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        self.model_path = rospy.get_param('~model_path', '/ros_ws/src/car/models/yolov8n.pt')

        self.imgsz = rospy.get_param(
            "~imgsz",
            640
        )

        self.inference_every_n_frames = rospy.get_param(
            "~inference_every_n_frames",
            1
        )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processing = False

        self.model = YOLO(self.model_path)

        self.image_pub = rospy.Publisher(
            "/yolo/image",
            Image,
            queue_size=1
        )
        self.mask_pub = rospy.Publisher(
            "/yolo/dynamic_mask",
            Image,
            queue_size=1
        )
        self.detection_pub = rospy.Publisher(
            "/yolo/detections",
            String,
            queue_size=10
        )

        self.rgb_sub = message_filters.Subscriber(
            self.image_topic,
            Image,
            queue_size=1,
            buff_size=2**24
        )
        self.camera_info_sub = message_filters.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            queue_size=1
        )

    def iamges_callback(self, rgb_msg, camera_info_msg):
        if self.processing:
            return

        self.processing = True
        self.frame_count += 1

        if self.frame_count % self.inference_every_n_frames != 0:
            self.processing = False
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS Image message to OpenCV image: {e}")
            self.processing = False
            return

        try:
            results = self.model(cv_image, imgsz=self.imgsz)
            detections = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            detection_list = []
            for bbox, conf, cls_id in zip(detections, confidences, class_ids):
                detection_list.append({
                    "bbox": bbox.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id)
                })

            detection_msg = String()
            detection_msg.data = json.dumps(detection_list)
            self.detection_pub.publish(detection_msg)

            annotated_image = results[0].plot()
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.image_pub.publish(annotated_image_msg)

        except Exception as e:
            rospy.logerr(f"Error during YOLO inference: {e}\n{traceback.format_exc()}")

        finally:
            self.processing = False


if __name__ == "__main__":

    try:

        node = CameraToYOLO()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass