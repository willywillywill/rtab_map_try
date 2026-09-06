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
import struct
import rospy
import message_filters
import rospkg
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
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
        self.scale_factor = rospy.get_param("~scale_factor", 1.0)
        self.jpeg_quality = rospy.get_param("~jpeg_quality", 80)

        # RGB
        self.image_topic = rospy.get_param(
            "~image_topic",
            "/camera/color/image_raw"
        )
        # Depth
        self.depth_topic = rospy.get_param(
            "~depth_topic",
            "/camera/depth/image_raw"
        )

        # YOLO model
        self.model_path = rospy.get_param(
            "~model",
            "/ros_ws/src/car/models/yolo26n-seg_ncnn_model"
        )

        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        self.filtered_rgb_pub = rospy.Publisher(
            "/camera/color/image_filtered", 
            Image,
            queue_size=1
            )
        self.filtered_depth_pub = rospy.Publisher(
                "/camera/depth/image_filtered", 
                Image, 
                queue_size=1
            )
        # 膨脹 Kernel (新增：避免動態物體邊緣洩漏)
        self.dilate_kernel = np.ones((9, 9), np.uint8)

        # 修改：使用 message_filters 同步訂閱 RGB 與 Depth
        self.rgb_sub = message_filters.Subscriber(self.image_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=5, 
            slop=0.05
        )
        self.sync.registerCallback(self.image_callback)


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

    @torch.no_grad()
    def image_callback(self, rgb_msg, depth_msg):
        if self.processing:
            return

        self.frame_count += 1
        if self.frame_count % self.inference_every_n_frames != 0:
            return

        self.processing = True

        try:
            # 1. 轉為 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            h, w = cv_image.shape[:2]

            # 2. YOLO 推論
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                verbose=False
            )[0]

            combined_mask = np.zeros((h, w), dtype=np.uint8)

            # 3. 處理分割遮罩 (Segmentation Mask)
            if results.masks is not None and results.boxes is not None:
                masks = results.masks.data.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                names = self.model.names

                for mask, cls_id in zip(masks, classes):
                    class_name = names.get(int(cls_id), "")
                    if class_name not in self.dynamic_classes:
                        continue

                    # 將 Mask 縮放到原圖尺寸
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask[mask_resized > 0.5] = 255

            # 4. 遮罩膨脹處理
            if np.any(combined_mask > 0):
                combined_mask = cv2.dilate(combined_mask, self.dilate_kernel, iterations=1)

            dynamic_area = combined_mask > 0

            # 5. 過濾動態區域 (塗黑)
            filtered_rgb = cv_image.copy()
            filtered_depth = cv_depth.copy()

            filtered_rgb[dynamic_area] = 0
            filtered_depth[dynamic_area] = 0

            # 6. 發布 RGB 圖像 (未壓縮 Image 格式)
            filtered_rgb = np.ascontiguousarray(filtered_rgb, dtype=np.uint8)
            
            # 若通道數不為 3，強制轉為 BGR
            if len(filtered_rgb.shape) == 2:
                filtered_rgb = cv2.cvtColor(filtered_rgb, cv2.COLOR_GRAY2BGR)
            elif filtered_rgb.shape[2] == 4:
                filtered_rgb = cv2.cvtColor(filtered_rgb, cv2.COLOR_BGRA2BGR)

            out_rgb_msg = self.numpy_to_ros_image(filtered_rgb, rgb_msg.header, "bgr8")
            out_rgb_msg.header = rgb_msg.header
            self.filtered_rgb_pub.publish(out_rgb_msg)

            # 7. 發布 Depth 圖像
            out_depth_msg = self.bridge.cv2_to_imgmsg(filtered_depth, encoding=depth_msg.encoding)
            out_depth_msg.header = depth_msg.header
            self.filtered_depth_pub.publish(out_depth_msg)

        except Exception as e:
            rospy.logerr("YOLO Segmentation Error: %s", str(e))
            rospy.logerr(traceback.format_exc())

        finally:
            self.processing = False

if __name__ == "__main__":
    try:
        node = YoloSegNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass