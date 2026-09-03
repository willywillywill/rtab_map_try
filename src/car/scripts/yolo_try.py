#!/opt/yolo_env/bin/python3

# ============================================================
# PyTorch / YOLO
# ============================================================
import ncnn
import torch
from ultralytics import YOLO


# ============================================================
# Python
# ============================================================

import json
import traceback

import cv2
import numpy as np


# ============================================================
# ROS
# ============================================================

import rospy
import message_filters
import rospkg
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError


class YoloSegNode:
    def __init__(self):
        rospy.init_node("yolo_seg_node", anonymous=False)
        
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)
        self.processing = False
        self.frame_count = 0
        self.inference_every_n_frames = rospy.get_param(
            "~inference_every_n_frames",
            1
        )

        # RGB
        self.image_topic = rospy.get_param(
            "~image_topic",
            "/camera/color/image_raw"
        )

        # YOLO model
        self.model_path = rospy.get_param(
            "~model",
            "car/models/yolo26n-seg_ncnn_model"
        )

        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            self.image_topic, 
            Image, 
            self.image_callback, 
            queue_size=1
        )

        self.mask_pub = rospy.Publisher("~segment_mask", Image, queue_size=1)
        self.overlay_pub = rospy.Publisher("~overlay_image", Image, queue_size=1)

        self.dynamic_classes = rospy.get_param(
            "~dynamic_classes",
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck",
                "dog"
            ]
        )


    def numpy_to_ros_image(
        self,
        image,
        header,
        encoding
    ):

        image = np.ascontiguousarray(
            image,
            dtype=np.uint8
        )

        ros_msg = Image()

        ros_msg.header = header

        ros_msg.height = image.shape[0]
        ros_msg.width = image.shape[1]

        ros_msg.encoding = encoding

        ros_msg.is_bigendian = 0

        # RGB / BGR
        if len(image.shape) == 3:

            channels = image.shape[2]

        # mono
        else:

            channels = 1

        ros_msg.step = (
            image.shape[1]
            * channels
        )

        ros_msg.data = image.tobytes()

        return ros_msg

    def image_callback(self, msg):
        if self.processing:
            return
        self.frame_count += 1

        if (
            self.frame_count
            % self.inference_every_n_frames
            != 0
        ):

            return
        self.processing = True

        try:

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


            results = self.model.predict(
                cv_image, 
                conf=self.conf_threshold, 
                verbose=False)[0]

            h, w, _ = cv_image.shape
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            overlay = cv_image.copy()

            if results.masks is not None and results.boxes is not None:
                masks = results.masks.data.cpu().numpy()

                classes = results.boxes.cls.cpu().numpy()
                names = self.model.names

                for mask, cls_id in zip(masks,classes):
                    class_name = names.get(int(cls_id), "")
                    if class_name not in self.dynamic_classes:
                        continue
                        
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0.5).astype(bool)

                    # 累加至總遮罩 (二值化：255)
                    combined_mask[binary_mask] = 255

                    # 半透明塗層繪製 (安全寫法：純 NumPy 加權算術)
                    green_layer = np.zeros_like(cv_image, dtype=np.uint8)
                    green_layer[:, :] = [0, 255, 0]  # BGR 綠色

                    # 將遮罩區域混合：50% 原圖 + 50% 綠色
                    overlay[binary_mask] = (
                        overlay[binary_mask].astype(np.float32) * 0.5 + 
                        green_layer[binary_mask].astype(np.float32) * 0.5
                    ).astype(np.uint8)

            
            # 強制整理記憶體佈局，防止 CvBridge 拋出 KeyError
            overlay_clean = np.ascontiguousarray(overlay, dtype=np.uint8)
            mask_clean = np.ascontiguousarray(combined_mask, dtype=np.uint8)
            mask_msg = self.numpy_to_ros_image(mask_clean, msg.header, "mono8")
            self.mask_pub.publish(mask_msg)
            
            overlay_msg = self.numpy_to_ros_image(overlay_clean, msg.header, encoding="bgr8")
            self.overlay_pub.publish(overlay_msg)
        except Exception as e:
            rospy.logerr(
                "YOLO error: %s",
                str(e)
            )

            rospy.logerr(
                traceback.format_exc()
            )

        finally:
            self.processing = False

if __name__ == "__main__":
    try:
        node = YoloSegNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass