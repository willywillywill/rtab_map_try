#!/opt/yolo_env/bin/python3
# -*- coding: utf-8 -*-

"""
camera_to_yolo - Raspberry Pi 5 / ROS Noetic optimized YOLO node
"""

import json
import math
import threading
import time
import traceback
import ncnn

import cv2
import numpy as np
import rospy
import message_filters
import torch

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from ultralytics import YOLO


class CameraToYOLO:
    def __init__(self):
        rospy.init_node("camera_to_yolo", anonymous=False)

        # ----------------------------
        # ROS parameters
        # ----------------------------
        self.image_topic = rospy.get_param(
            "~image_topic", "/camera/color/image_raw"
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/depth/image_raw"
        )
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )

        self.model_path = rospy.get_param(
            "~model",
            "/ros_ws/src/car/models/yolov8n_ncnn_model"
        )

        self.conf = float(rospy.get_param("~conf", 0.45))
        self.iou = float(rospy.get_param("~iou", 0.50))
        self.imgsz = int(rospy.get_param("~imgsz", 320))
        self.device = rospy.get_param("~device", "cpu")

        self.inference_every_n_frames = max(
            1, int(rospy.get_param("~inference_every_n_frames", 1))
        )

        self.max_det = int(rospy.get_param("~max_det", 30))

        self.publish_debug_image = bool(
            rospy.get_param("~publish_debug_image", True)
        )
        self.draw_labels = bool(
            rospy.get_param("~draw_labels", True)
        )

        self.publish_mask = bool(
            rospy.get_param("~publish_mask", True)
        )

        self.output_max_fps = float(
            rospy.get_param("~output_max_fps", 0.0)
        )

        # ----------------------------
        # Depth parameters
        # ----------------------------
        self.depth_center_ratio = float(
            rospy.get_param("~depth_center_ratio", 0.40)
        )
        self.min_depth = float(
            rospy.get_param("~min_depth", 0.15)
        )
        self.max_depth = float(
            rospy.get_param("~max_depth", 8.0)
        )
        self.depth_sample_step = max(
            1, int(rospy.get_param("~depth_sample_step", 2))
        )

        # ----------------------------
        # Dynamic classes
        # ----------------------------
        default_dynamic = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "dog",
        ]
        self.dynamic_classes = set(
            rospy.get_param("~dynamic_classes", default_dynamic)
        )

        self.classes = rospy.get_param("~classes", [])
        if self.classes is not None:
            self.classes = [int(x) for x in self.classes]

        # ----------------------------
        # Runtime state
        # ----------------------------
        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_header = None
        self.new_frame_id = 0
        self.last_processed_frame_id = -1

        self.frame_counter = 0
        self.inference_counter = 0

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_ready = False

        self.running = True
        self.last_publish_time = 0.0

        self.fps_window_start = time.perf_counter()
        self.fps_window_count = 0
        self.current_fps = 0.0
        self.last_inference_ms = 0.0

        # ----------------------------
        # CPU Tuning
        # ----------------------------
        torch_threads = int(rospy.get_param("~torch_threads", 4))
        torch_threads = max(1, torch_threads)

        try:
            torch.set_num_threads(torch_threads)
            rospy.loginfo("PyTorch CPU threads: %d", torch_threads)
        except Exception as exc:
            rospy.logwarn("Could not set torch threads: %s", exc)

        cv_threads = int(rospy.get_param("~opencv_threads", 2))
        try:
            cv2.setNumThreads(max(0, cv_threads))
            rospy.loginfo("OpenCV threads: %d", max(0, cv_threads))
        except Exception:
            pass

        # ----------------------------
        # Load YOLO
        # ----------------------------
        rospy.loginfo("Loading YOLO model: %s", self.model_path)

        try:
            self.model = YOLO(self.model_path)
        except Exception:
            rospy.logerr("Failed to load YOLO model.")
            rospy.logerr(traceback.format_exc())
            raise

        rospy.loginfo("YOLO model loaded.")

        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            rospy.loginfo("Warming up YOLO...")
            self.model.predict(
                source=dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                max_det=self.max_det,
            )
            rospy.loginfo("YOLO warm-up complete.")
        except Exception as exc:
            rospy.logwarn("YOLO warm-up failed: %s", exc)

        # ----------------------------
        # Publishers
        # ----------------------------
        self.image_pub = rospy.Publisher(
            "/yolo/image", Image, queue_size=1
        )
        self.mask_pub = rospy.Publisher(
            "/yolo/dynamic_mask", Image, queue_size=1
        )
        self.detection_pub = rospy.Publisher(
            "/yolo/detections", String, queue_size=2
        )
        self.fps_pub = rospy.Publisher(
            "/yolo/fps", Float32, queue_size=1
        )

        # ----------------------------
        # Subscribers
        # ----------------------------
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )

        rgb_sub = message_filters.Subscriber(
            self.image_topic, Image
        )
        depth_sub = message_filters.Subscriber(
            self.depth_topic, Image
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=2,
            slop=0.08,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        worker_hz = float(rospy.get_param("~worker_hz", 30.0))
        worker_hz = max(1.0, worker_hz)

        self.worker = threading.Thread(
            target=self.inference_worker,
            name="yolo_worker",
            daemon=True,
        )
        self.worker.start()

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / worker_hz),
            self.worker_tick,
        )

        rospy.loginfo("camera_to_yolo node running.")

    def camera_info_callback(self, msg):
        if len(msg.K) < 9:
            return

        fx = float(msg.K[0])
        fy = float(msg.K[4])
        cx = float(msg.K[2])
        cy = float(msg.K[5])

        if fx <= 0.0 or fy <= 0.0:
            return

        with self.lock:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.camera_info_ready = True

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        self.frame_counter += 1

        if (
            self.frame_counter % self.inference_every_n_frames
            != 0
        ):
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8",
            )
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            with self.lock:
                self.latest_rgb = rgb
                self.latest_depth = depth
                self.latest_header = rgb_msg.header
                self.new_frame_id += 1

        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "CvBridge RGB-D error: %s", str(exc))
        except Exception:
            rospy.logerr_throttle(2.0, "RGB-D callback error:\n%s", traceback.format_exc())

    def worker_tick(self, _event):
        pass

    def get_latest_frame(self):
        with self.lock:
            if self.latest_rgb is None:
                return None

            if self.new_frame_id == self.last_processed_frame_id:
                return None

            rgb = self.latest_rgb
            depth = self.latest_depth
            header = self.latest_header
            frame_id = self.new_frame_id

            self.last_processed_frame_id = frame_id

        return rgb, depth, header

    def inference_worker(self):
        while not rospy.is_shutdown() and self.running:
            packet = self.get_latest_frame()

            if packet is None:
                time.sleep(0.001)
                continue

            rgb, depth, header = packet

            try:
                self.process_frame(rgb, depth, header)
            except Exception:
                rospy.logerr_throttle(
                    2.0,
                    "YOLO worker error:\n%s",
                    traceback.format_exc(),
                )

    def process_frame(self, frame, depth_image, header):
        t0 = time.perf_counter()
        height, width = frame.shape[:2]

        predict_kwargs = dict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            max_det=self.max_det,
        )

        if self.classes:
            predict_kwargs["classes"] = self.classes

        results = self.model.predict(**predict_kwargs)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = inference_ms

        if not results:
            return

        result = results[0]

        dynamic_mask = np.zeros((height, width), dtype=np.uint8)
        annotated_frame = frame.copy() if self.publish_debug_image else None
        detections = []

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]

                x1 = max(0, min(width - 1, int(x1)))
                y1 = max(0, min(height - 1, int(y1)))
                x2 = max(0, min(width - 1, int(x2)))
                y2 = max(0, min(height - 1, int(y2)))

                if x2 <= x1 or y2 <= y1:
                    continue

                class_id = int(class_ids[i])
                confidence = float(confs[i])

                class_name = self.model.names.get(
                    class_id, str(class_id)
                )

                is_dynamic = class_name in self.dynamic_classes

                center_u = (x1 + x2) // 2
                center_v = (y1 + y2) // 2

                depth_m = self.get_depth_median(
                    depth_image,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                position_3d = None
                distance_3d = None

                if depth_m is not None:
                    position_3d = self.project_pixel_to_3d(
                        center_u,
                        center_v,
                        depth_m,
                    )

                    if position_3d is not None:
                        X, Y, Z = position_3d
                        distance_3d = math.sqrt(
                            X * X + Y * Y + Z * Z
                        )

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "center_pixel": {
                        "u": center_u,
                        "v": center_v,
                    },
                    "potential_dynamic": is_dynamic,
                    "depth_m": (
                        round(depth_m, 4)
                        if depth_m is not None
                        else None
                    ),
                    "distance_3d_m": (
                        round(distance_3d, 4)
                        if distance_3d is not None
                        else None
                    ),
                }

                if position_3d is not None:
                    X, Y, Z = position_3d
                    detection["position_camera"] = {
                        "x": round(X, 4),
                        "y": round(Y, 4),
                        "z": round(Z, 4),
                    }
                else:
                    detection["position_camera"] = None

                detections.append(detection)

                if is_dynamic and self.publish_mask:
                    cv2.rectangle(
                        dynamic_mask,
                        (x1, y1),
                        (x2, y2),
                        255,
                        thickness=-1,
                    )

                if annotated_frame is not None:
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.circle(
                        annotated_frame,
                        (center_u, center_v),
                        3,
                        (0, 0, 255),
                        -1,
                    )

                    if self.draw_labels:
                        label = (
                            f"{class_name} {confidence:.2f} Z:{depth_m:.2f}m"
                            if depth_m is not None
                            else f"{class_name} {confidence:.2f} Depth:N/A"
                        )

                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

        # --------------------------------------------------------
        # Publish
        # --------------------------------------------------------
        now = time.perf_counter()

        if (
            self.output_max_fps > 0.0
            and self.last_publish_time > 0.0
        ):
            min_period = 1.0 / self.output_max_fps
            if now - self.last_publish_time < min_period:
                self.publish_detections(detections, header)
                self.update_fps()
                return

        self.last_publish_time = now

        # Explicit C-contiguous uint8 cast prevents CvBridge KeyError: 16
        if annotated_frame is not None:
            clean_frame = np.ascontiguousarray(annotated_frame, dtype=np.uint8)
            image_msg = self.bridge.cv2_to_imgmsg(
                clean_frame,
                encoding="bgr8",
            )
            image_msg.header = header
            self.image_pub.publish(image_msg)

        if self.publish_mask:
            clean_mask = np.ascontiguousarray(dynamic_mask, dtype=np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(
                clean_mask,
                encoding="mono8",
            )
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

        self.publish_detections(detections, header)

        self.inference_counter += 1
        self.update_fps()

    def get_depth_median(self, depth_image, x1, y1, x2, y2):
        h, w = depth_image.shape[:2]

        ratio = min(max(self.depth_center_ratio, 0.1), 1.0)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        roi_w = max(1, int(bw * ratio))
        roi_h = max(1, int(bh * ratio))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        rx1 = max(0, cx - roi_w // 2)
        ry1 = max(0, cy - roi_h // 2)
        rx2 = min(w, cx + roi_w // 2)
        ry2 = min(h, cy + roi_h // 2)

        roi = np.copy(
            depth_image[
                ry1:ry2:self.depth_sample_step,
                rx1:rx2:self.depth_sample_step,
            ]
        )

        if roi.size == 0:
            return None

        if roi.dtype == np.uint16:
            valid = roi[
                (roi > 0) &
                (roi >= int(self.min_depth * 1000.0)) &
                (roi <= int(self.max_depth * 1000.0))
            ]

            if valid.size == 0:
                return None

            return float(np.median(valid)) * 0.001

        if roi.dtype in (np.float32, np.float64):
            valid = roi[
                np.isfinite(roi) &
                (roi >= self.min_depth) &
                (roi <= self.max_depth)
            ]

            if valid.size == 0:
                return None

            return float(np.median(valid))

        return None

    def project_pixel_to_3d(self, u, v, depth_m):
        with self.lock:
            if not self.camera_info_ready:
                return None

            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy

        if fx is None or fy is None:
            return None

        Z = float(depth_m)

        if Z <= 0.0:
            return None

        X = (float(u) - cx) * Z / fx
        Y = (float(v) - cy) * Z / fy

        return X, Y, Z

    def publish_detections(self, detections, header):
        msg = {
            "stamp": header.stamp.to_sec(),
            "frame_id": header.frame_id,
            "objects": detections,
        }

        self.detection_pub.publish(
            json.dumps(
                msg,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def update_fps(self):
        self.fps_window_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_window_start

        if elapsed >= 1.0:
            self.current_fps = self.fps_window_count / elapsed
            self.fps_pub.publish(Float32(self.current_fps))
            self.fps_window_start = now
            self.fps_window_count = 0

    def shutdown(self):
        self.running = False


def main():
    try:
        node = CameraToYOLO()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception:
        rospy.logerr("camera_to_yolo fatal error:\n%s", traceback.format_exc())


if __name__ == "__main__":
    main()