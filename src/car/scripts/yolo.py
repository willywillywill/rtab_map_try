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

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError


# ============================================================
# Camera -> YOLO -> Depth -> 3D
# ============================================================

class CameraToYOLO:

    def __init__(self):

        rospy.init_node(
            "camera_to_yolo",
            anonymous=False
        )

        # ====================================================
        # ROS Parameters
        # ====================================================

        # RGB
        self.image_topic = rospy.get_param(
            "~image_topic",
            "/camera/color/image_raw"
        )

        # Depth
        #
        # 如果 Astra 有：
        # /camera/depth_registered/image_raw
        #
        # 建議優先使用 registered depth
        self.depth_topic = rospy.get_param(
            "~depth_topic",
            "/camera/depth/image_raw"
        )

        # 使用 RGB CameraInfo
        #
        # 因為 YOLO bbox 是基於 RGB image
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            "/camera/color/camera_info"
        )

        # YOLO model
        self.model_path = rospy.get_param(
            "~model",
            "/ros_ws/src/car/models/yolov8n.pt"
        )

        self.conf = rospy.get_param(
            "~conf",
            0.5
        )

        self.iou = rospy.get_param(
            "~iou",
            0.45
        )

        self.imgsz = rospy.get_param(
            "~imgsz",
            640
        )

        # CPU:
        #   cpu
        #
        # NVIDIA GPU:
        #   0
        self.device = rospy.get_param(
            "~device",
            "cpu"
        )

        # 每幾張圖做一次 inference
        self.inference_every_n_frames = rospy.get_param(
            "~inference_every_n_frames",
            1
        )

        # ====================================================
        # Depth Parameters
        # ====================================================

        # bbox 中央多少比例拿來算深度
        #
        # 0.4 = 中央 40%
        self.depth_center_ratio = rospy.get_param(
            "~depth_center_ratio",
            0.4
        )

        # Depth 最小有效距離
        self.min_depth = rospy.get_param(
            "~min_depth",
            0.15
        )

        # Depth 最大有效距離
        self.max_depth = rospy.get_param(
            "~max_depth",
            8.0
        )

        # ====================================================
        # Potential Dynamic Classes
        # ====================================================

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

        # ====================================================
        # Runtime
        # ====================================================

        self.bridge = CvBridge()

        self.frame_count = 0
        self.processing = False

        # ====================================================
        # Load YOLO
        # ====================================================

        rospy.loginfo(
            "Loading YOLO model: %s",
            self.model_path
        )

        self.model = YOLO(
            self.model_path
        )

        rospy.loginfo(
            "YOLO model loaded."
        )

        # ====================================================
        # Publishers
        # ====================================================

        # YOLO result image
        self.image_pub = rospy.Publisher(
            "/yolo/image",
            Image,
            queue_size=1
        )

        # Potential Dynamic Mask
        self.mask_pub = rospy.Publisher(
            "/yolo/dynamic_mask",
            Image,
            queue_size=1
        )

        # Detection + XYZ JSON
        self.detection_pub = rospy.Publisher(
            "/yolo/detections",
            String,
            queue_size=10
        )

        # ====================================================
        # Subscribers
        #
        # RGB + Depth + CameraInfo synchronization
        # ====================================================

        self.rgb_sub = message_filters.Subscriber(
            self.image_topic,
            Image,
            queue_size=1,
            buff_size=2**24
        )

        self.depth_sub = message_filters.Subscriber(
            self.depth_topic,
            Image,
            queue_size=1,
            buff_size=2**24
        )

        self.camera_info_sub = message_filters.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            queue_size=1
        )

        # Approximate synchronization
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [
                self.rgb_sub,
                self.depth_sub,
                self.camera_info_sub
            ],
            queue_size=10,
            slop=0.15
        )

        self.sync.registerCallback(
            self.image_callback
        )

        # ====================================================
        # Startup Info
        # ====================================================

        rospy.loginfo(
            "========================================"
        )

        rospy.loginfo(
            "Camera -> YOLO -> Depth -> XYZ started"
        )

        rospy.loginfo(
            "RGB topic       : %s",
            self.image_topic
        )

        rospy.loginfo(
            "Depth topic     : %s",
            self.depth_topic
        )

        rospy.loginfo(
            "CameraInfo topic: %s",
            self.camera_info_topic
        )

        rospy.loginfo(
            "Model           : %s",
            self.model_path
        )

        rospy.loginfo(
            "Device          : %s",
            self.device
        )

        rospy.loginfo(
            "Confidence      : %.2f",
            self.conf
        )

        rospy.loginfo(
            "Depth ROI ratio : %.2f",
            self.depth_center_ratio
        )

        rospy.loginfo(
            "========================================"
        )

    # ========================================================
    # ROS Image -> Depth numpy
    # ========================================================

    def get_depth_image(
        self,
        depth_msg
    ):

        """
        ROS Depth Image -> float32 depth image (meter)

        常見 encoding：

        16UC1
            單位通常是 mm

        32FC1
            單位通常是 meter
        """

        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough"
        )

        if depth_msg.encoding in [
            "16UC1",
            "mono16"
        ]:

            # mm -> m
            depth_image = (
                depth_image.astype(
                    np.float32
                )
                / 1000.0
            )

        elif depth_msg.encoding == "32FC1":

            depth_image = depth_image.astype(
                np.float32
            )

        else:

            rospy.logwarn_throttle(
                5.0,
                "Unknown depth encoding: %s",
                depth_msg.encoding
            )

            depth_image = depth_image.astype(
                np.float32
            )

        return depth_image

    # ========================================================
    # Get bbox center depth
    # ========================================================

    def get_bbox_depth(
        self,
        depth_image,
        x1,
        y1,
        x2,
        y2
    ):

        """
        取得 YOLO bbox 中央 ROI 的 median depth。

        為什麼不用正中央單一 pixel？

        因為單一 pixel 可能：
        - depth = 0
        - sensor noise
        - reflective surface
        - depth hole

        因此使用中央 ROI + median 比較穩定。
        """

        height, width = depth_image.shape[:2]

        # --------------------------------------------
        # bbox center
        # --------------------------------------------

        center_u = int(
            (x1 + x2) / 2
        )

        center_v = int(
            (y1 + y2) / 2
        )

        bbox_width = max(
            1,
            x2 - x1
        )

        bbox_height = max(
            1,
            y2 - y1
        )

        # --------------------------------------------
        # ROI size
        # --------------------------------------------

        roi_width = max(
            2,
            int(
                bbox_width
                * self.depth_center_ratio
            )
        )

        roi_height = max(
            2,
            int(
                bbox_height
                * self.depth_center_ratio
            )
        )

        roi_x1 = max(
            0,
            center_u - roi_width // 2
        )

        roi_x2 = min(
            width,
            center_u + roi_width // 2
        )

        roi_y1 = max(
            0,
            center_v - roi_height // 2
        )

        roi_y2 = min(
            height,
            center_v + roi_height // 2
        )

        # --------------------------------------------
        # invalid ROI
        # --------------------------------------------

        if (
            roi_x2 <= roi_x1
            or
            roi_y2 <= roi_y1
        ):

            return (
                None,
                center_u,
                center_v,
                None
            )

        # --------------------------------------------
        # Crop Depth ROI
        # --------------------------------------------

        depth_roi = depth_image[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        # --------------------------------------------
        # Valid Depth
        # --------------------------------------------

        valid_mask = (
            np.isfinite(
                depth_roi
            )
            &
            (
                depth_roi
                >= self.min_depth
            )
            &
            (
                depth_roi
                <= self.max_depth
            )
        )

        valid_depth = depth_roi[
            valid_mask
        ]

        # 沒有有效 Depth
        if valid_depth.size == 0:

            return (
                None,
                center_u,
                center_v,
                (
                    roi_x1,
                    roi_y1,
                    roi_x2,
                    roi_y2
                )
            )

        # --------------------------------------------
        # Median Depth
        # --------------------------------------------

        depth_meter = float(
            np.median(
                valid_depth
            )
        )

        return (
            depth_meter,
            center_u,
            center_v,
            (
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2
            )
        )

    # ========================================================
    # Pixel + Depth -> XYZ
    # ========================================================

    def pixel_to_3d(
        self,
        u,
        v,
        depth,
        camera_info
    ):

        """
        Pixel coordinate (u, v)
        + Depth Z
        + Camera Intrinsics
        -> Camera Optical Frame XYZ

        X = (u-cx) * Z / fx
        Y = (v-cy) * Z / fy
        Z = depth

        ROS Camera Optical Frame:

        X = right
        Y = down
        Z = forward
        """

        fx = camera_info.K[0]
        fy = camera_info.K[4]

        cx = camera_info.K[2]
        cy = camera_info.K[5]

        if (
            fx == 0
            or
            fy == 0
        ):

            rospy.logwarn_throttle(
                5.0,
                "Invalid CameraInfo: fx or fy = 0"
            )

            return None

        Z = float(
            depth
        )

        X = (
            (
                float(u)
                - cx
            )
            * Z
            / fx
        )

        Y = (
            (
                float(v)
                - cy
            )
            * Z
            / fy
        )

        return (
            float(X),
            float(Y),
            float(Z)
        )

    # ========================================================
    # Create ROS Image manually
    #
    # 避免 cv_bridge.cv2_to_imgmsg()
    # 與 pip OpenCV / ROS OpenCV ABI 衝突
    # ========================================================

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

    # ========================================================
    # Main Callback
    # ========================================================

    def image_callback(
        self,
        rgb_msg,
        depth_msg,
        camera_info_msg
    ):

        # --------------------------------------------
        # 防止上一張還沒處理完
        # --------------------------------------------

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

            # =================================================
            # RGB -> OpenCV
            # =================================================

            frame = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8"
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # =================================================
            # Depth -> meter
            # =================================================

            depth_image = self.get_depth_image(
                depth_msg
            )

            depth_height, depth_width = (
                depth_image.shape[:2]
            )

            # =================================================
            # Check RGB / Depth alignment
            # =================================================

            if (
                frame_width != depth_width
                or
                frame_height != depth_height
            ):

                rospy.logwarn_throttle(
                    5.0,
                    (
                        "RGB size=%dx%d "
                        "Depth size=%dx%d. "
                        "Depth may not be aligned."
                    ),
                    frame_width,
                    frame_height,
                    depth_width,
                    depth_height
                )

            # =================================================
            # YOLO Inference
            # =================================================

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            result = results[0]

            # =================================================
            # Potential Dynamic Mask
            # =================================================

            dynamic_mask = np.zeros(
                (
                    frame_height,
                    frame_width
                ),
                dtype=np.uint8
            )

            # =================================================
            # Detection Results
            # =================================================

            detections = []

            # =================================================
            # YOLO Boxes
            # =================================================

            if result.boxes is not None:

                for i, box in enumerate(
                    result.boxes
                ):

                    # -----------------------------------------
                    # class id
                    # -----------------------------------------

                    class_id = int(
                        box.cls[0].item()
                    )

                    # -----------------------------------------
                    # class name
                    # -----------------------------------------

                    class_name = (
                        self.model.names[
                            class_id
                        ]
                    )

                    # -----------------------------------------
                    # confidence
                    # -----------------------------------------

                    confidence = float(
                        box.conf[0].item()
                    )

                    # -----------------------------------------
                    # Bounding Box
                    # -----------------------------------------

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    x1 = int(
                        max(
                            0,
                            x1
                        )
                    )

                    y1 = int(
                        max(
                            0,
                            y1
                        )
                    )

                    x2 = int(
                        min(
                            frame_width - 1,
                            x2
                        )
                    )

                    y2 = int(
                        min(
                            frame_height - 1,
                            y2
                        )
                    )

                    # -----------------------------------------
                    # Potential Dynamic
                    # -----------------------------------------

                    is_potential_dynamic = (
                        class_name
                        in self.dynamic_classes
                    )

                    # =========================================
                    # Depth
                    # =========================================

                    (
                        depth_meter,
                        center_u,
                        center_v,
                        depth_roi
                    ) = self.get_bbox_depth(
                        depth_image,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    # =========================================
                    # XYZ
                    # =========================================

                    position_3d = None

                    distance_3d = None

                    if depth_meter is not None:

                        position_3d = (
                            self.pixel_to_3d(
                                center_u,
                                center_v,
                                depth_meter,
                                camera_info_msg
                            )
                        )

                        if position_3d is not None:

                            X, Y, Z = (
                                position_3d
                            )

                            distance_3d = float(
                                np.sqrt(
                                    X * X
                                    +
                                    Y * Y
                                    +
                                    Z * Z
                                )
                            )

                    # =========================================
                    # Detection JSON
                    # =========================================

                    detection = {

                        "class_id":
                            class_id,

                        "class_name":
                            class_name,

                        "confidence":
                            confidence,

                        "bbox": {

                            "x1": x1,
                            "y1": y1,

                            "x2": x2,
                            "y2": y2

                        },

                        "center_pixel": {

                            "u": center_u,
                            "v": center_v

                        },

                        "potential_dynamic":
                            is_potential_dynamic,

                        "depth_m":
                            depth_meter,

                        "distance_3d_m":
                            distance_3d,

                        "position_camera":
                            None
                    }

                    # =========================================
                    # Store XYZ
                    # =========================================

                    if position_3d is not None:

                        X, Y, Z = (
                            position_3d
                        )

                        detection[
                            "position_camera"
                        ] = {

                            "x": X,
                            "y": Y,
                            "z": Z

                        }

                    detections.append(
                        detection
                    )

                    # =========================================
                    # Dynamic Mask
                    # =========================================

                    if is_potential_dynamic:

                        # -------------------------------------
                        # YOLO Segmentation
                        # -------------------------------------

                        if (
                            result.masks is not None
                            and
                            result.masks.data
                            is not None
                            and
                            i < len(
                                result.masks.data
                            )
                        ):

                            mask = (
                                result.masks.data[i]
                                .cpu()
                                .numpy()
                            )

                            mask = cv2.resize(
                                mask,
                                (
                                    frame_width,
                                    frame_height
                                ),
                                interpolation=
                                cv2.INTER_NEAREST
                            )

                            binary_mask = (
                                mask > 0.5
                            ).astype(
                                np.uint8
                            ) * 255

                            dynamic_mask = (
                                cv2.bitwise_or(
                                    dynamic_mask,
                                    binary_mask
                                )
                            )

                        # -------------------------------------
                        # Detection Model
                        #
                        # Bounding box -> Mask
                        # -------------------------------------

                        else:

                            cv2.rectangle(
                                dynamic_mask,
                                (
                                    x1,
                                    y1
                                ),
                                (
                                    x2,
                                    y2
                                ),
                                255,
                                thickness=-1
                            )

            # =================================================
            # YOLO Plot
            # =================================================

            annotated_frame = (
                result.plot()
            )

            # =================================================
            # Draw Depth / XYZ information
            # =================================================

            for detection in detections:

                bbox = detection[
                    "bbox"
                ]

                x1 = bbox["x1"]
                y1 = bbox["y1"]

                class_name = detection[
                    "class_name"
                ]

                depth_meter = detection[
                    "depth_m"
                ]

                position = detection[
                    "position_camera"
                ]

                distance_3d = detection[
                    "distance_3d_m"
                ]

                # -----------------------------------------
                # center point
                # -----------------------------------------

                center_u = detection[
                    "center_pixel"
                ]["u"]

                center_v = detection[
                    "center_pixel"
                ]["v"]

                cv2.circle(
                    annotated_frame,
                    (
                        center_u,
                        center_v
                    ),
                    4,
                    (
                        0,
                        0,
                        255
                    ),
                    -1
                )

                # -----------------------------------------
                # Depth valid
                # -----------------------------------------

                if (
                    depth_meter is not None
                    and
                    position is not None
                ):

                    X = position["x"]
                    Y = position["y"]
                    Z = position["z"]

                    # Distance
                    depth_text = (
                        f"{class_name} "
                        f"Z:{Z:.2f}m "
                        f"D:{distance_3d:.2f}m"
                    )

                    xyz_text = (
                        f"XYZ("
                        f"{X:.2f},"
                        f"{Y:.2f},"
                        f"{Z:.2f}"
                        f")"
                    )

                    cv2.putText(
                        annotated_frame,
                        depth_text,
                        (
                            x1,
                            max(
                                25,
                                y1 - 25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (
                            0,
                            255,
                            0
                        ),
                        1,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        annotated_frame,
                        xyz_text,
                        (
                            x1,
                            max(
                                15,
                                y1 - 8
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (
                            0,
                            255,
                            255
                        ),
                        1,
                        cv2.LINE_AA
                    )

                # -----------------------------------------
                # No depth
                # -----------------------------------------

                else:

                    cv2.putText(
                        annotated_frame,
                        (
                            f"{class_name} "
                            f"Depth:N/A"
                        ),
                        (
                            x1,
                            max(
                                20,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (
                            0,
                            0,
                            255
                        ),
                        1,
                        cv2.LINE_AA
                    )

            # =================================================
            # Publish Annotated Image
            # =================================================

            image_msg = (
                self.numpy_to_ros_image(
                    annotated_frame,
                    rgb_msg.header,
                    "bgr8"
                )
            )

            self.image_pub.publish(
                image_msg
            )

            # =================================================
            # Publish Dynamic Mask
            #
            # 手動建立 ROS Image，
            # 避免之前 cv_bridge KeyError:16
            # =================================================

            mask_msg = (
                self.numpy_to_ros_image(
                    dynamic_mask,
                    rgb_msg.header,
                    "mono8"
                )
            )

            self.mask_pub.publish(
                mask_msg
            )

            # =================================================
            # Publish JSON
            # =================================================

            detection_message = {

                "stamp":
                    rgb_msg.header.stamp.to_sec(),

                "frame_id":
                    rgb_msg.header.frame_id,

                "depth_frame_id":
                    depth_msg.header.frame_id,

                "objects":
                    detections

            }

            self.detection_pub.publish(
                json.dumps(
                    detection_message,
                    ensure_ascii=False
                )
            )

            # =================================================
            # Console
            # =================================================

            valid_depth_count = sum(
                1
                for detection
                in detections
                if detection[
                    "depth_m"
                ] is not None
            )

            dynamic_count = sum(
                1
                for detection
                in detections
                if detection[
                    "potential_dynamic"
                ]
            )

            rospy.loginfo_throttle(
                2.0,
                (
                    "YOLO objects=%d "
                    "depth_valid=%d "
                    "potential_dynamic=%d"
                ),
                len(
                    detections
                ),
                valid_depth_count,
                dynamic_count
            )

        # ====================================================
        # Exceptions
        # ====================================================

        except CvBridgeError as e:

            rospy.logerr(
                "CvBridge error: %s",
                str(e)
            )

            rospy.logerr(
                traceback.format_exc()
            )

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        node = CameraToYOLO()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass