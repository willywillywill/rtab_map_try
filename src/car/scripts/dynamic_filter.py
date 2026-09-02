#!/opt/yolo_env/bin/python3

# -*- coding: utf-8 -*-

"""
Dynamic PointCloud Filter
=========================

ROS Noetic node

Input:
    /camera/depth_registered/points
        sensor_msgs/PointCloud2

    /yolo/dynamic_mask
        sensor_msgs/Image (mono8)

Output:
    /camera/static_points
        sensor_msgs/PointCloud2

功能：
    1. 接收 Astra depth registered point cloud
    2. 接收 YOLO dynamic mask
    3. 將 PointCloud2 的 3D 點投影到 camera image
    4. 如果該 pixel 位於 YOLO dynamic mask：
           → 移除該 3D point
    5. 保留 static / non-dynamic points
    6. 發布新的 PointCloud2 給 RTAB-Map

注意：
    PointCloud2 與 dynamic_mask 必須對應到同一個 camera optical frame。
"""

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge


class DynamicPointCloudFilter:

    def __init__(self):

        rospy.init_node(
            "dynamic_pointcloud_filter",
            anonymous=False
        )

        # ============================================================
        # Parameters
        # ============================================================

        self.cloud_topic = rospy.get_param(
            "~cloud_topic",
            "/camera/depth_registered/points"
        )

        self.mask_topic = rospy.get_param(
            "~mask_topic",
            "/yolo/dynamic_mask"
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/camera/static_points"
        )

        # Camera intrinsics
        self.fx = rospy.get_param("~fx", 0.0)
        self.fy = rospy.get_param("~fy", 0.0)
        self.cx = rospy.get_param("~cx", 0.0)
        self.cy = rospy.get_param("~cy", 0.0)

        # Mask dilation
        self.mask_dilation = rospy.get_param(
            "~mask_dilation",
            5
        )

        # Minimum / maximum valid depth
        self.min_depth = rospy.get_param(
            "~min_depth",
            0.15
        )

        self.max_depth = rospy.get_param(
            "~max_depth",
            8.0
        )

        # Skip pixels to improve Raspberry Pi performance
        self.point_step = rospy.get_param(
            "~point_step",
            1
        )

        # ============================================================
        # ROS
        # ============================================================

        self.bridge = CvBridge()

        self.latest_mask = None
        self.latest_mask_stamp = None

        self.cloud_count = 0

        # Publisher
        self.cloud_pub = rospy.Publisher(
            self.output_topic,
            PointCloud2,
            queue_size=1
        )

        # Subscribers
        self.mask_sub = rospy.Subscriber(
            self.mask_topic,
            Image,
            self.mask_callback,
            queue_size=1,
            buff_size=2**24
        )

        self.cloud_sub = rospy.Subscriber(
            self.cloud_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo(
            "=============================================="
        )
        rospy.loginfo(
            "Dynamic PointCloud Filter Started"
        )
        rospy.loginfo(
            "Cloud : %s",
            self.cloud_topic
        )
        rospy.loginfo(
            "Mask  : %s",
            self.mask_topic
        )
        rospy.loginfo(
            "Output: %s",
            self.output_topic
        )
        rospy.loginfo(
            "=============================================="
        )

    # ================================================================
    # YOLO Dynamic Mask
    # ================================================================

    def mask_callback(self, msg):

        try:

            mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8"
            )

            self.latest_mask = mask
            self.latest_mask_stamp = msg.header.stamp

        except Exception as e:

            rospy.logwarn_throttle(
                5.0,
                "Failed to convert YOLO mask: %s",
                str(e)
            )

    # ================================================================
    # PointCloud Callback
    # ================================================================

    def cloud_callback(self, msg):

        if self.latest_mask is None:
            rospy.logwarn_throttle(
                5.0,
                "Waiting for YOLO dynamic mask..."
            )
            return

        # ------------------------------------------------------------
        # Check camera intrinsics
        # ------------------------------------------------------------

        if (
            self.fx <= 0.0 or
            self.fy <= 0.0
        ):

            rospy.logwarn_throttle(
                5.0,
                "Camera intrinsics are not configured."
            )

            return

        # ------------------------------------------------------------
        # Get mask
        # ------------------------------------------------------------

        mask = self.latest_mask

        height, width = mask.shape

        # ------------------------------------------------------------
        # Read PointCloud2
        #
        # skip_nans=True:
        # 不處理 NaN point
        # ------------------------------------------------------------

        try:

            points = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

        except Exception as e:

            rospy.logwarn_throttle(
                5.0,
                "Failed to read PointCloud2: %s",
                str(e)
            )

            return

        # ------------------------------------------------------------
        # Filtering
        # ------------------------------------------------------------

        static_points = []

        total_points = 0
        removed_points = 0

        step = max(1, int(self.point_step))

        for index, p in enumerate(points):

            # Downsample
            if index % step != 0:
                continue

            x = p[0]
            y = p[1]
            z = p[2]

            total_points += 1

            # --------------------------------------------------------
            # Validate XYZ
            # --------------------------------------------------------

            if not np.isfinite(x):
                continue

            if not np.isfinite(y):
                continue

            if not np.isfinite(z):
                continue

            if z < self.min_depth:
                continue

            if z > self.max_depth:
                continue

            # --------------------------------------------------------
            # Project 3D point to camera image
            #
            # Astra optical frame:
            #
            # X -> right
            # Y -> down
            # Z -> forward
            # --------------------------------------------------------

            if z <= 0.001:
                continue

            u = int(
                (x * self.fx / z) + self.cx
            )

            v = int(
                (y * self.fy / z) + self.cy
            )

            # --------------------------------------------------------
            # Check image boundary
            # --------------------------------------------------------

            if u < 0 or u >= width:
                continue

            if v < 0 or v >= height:
                continue

            # --------------------------------------------------------
            # YOLO dynamic mask
            #
            # mask = 0 -> static
            # mask > 0 -> dynamic
            # --------------------------------------------------------

            if mask[v, u] > 0:

                removed_points += 1

                continue

            # --------------------------------------------------------
            # Keep static point
            # --------------------------------------------------------

            static_points.append(
                (x, y, z)
            )

        # ============================================================
        # Create output PointCloud2
        # ============================================================

        header = msg.header

        output_cloud = point_cloud2.create_cloud_xyz32(
            header,
            static_points
        )

        # ============================================================
        # Publish
        # ============================================================

        self.cloud_pub.publish(
            output_cloud
        )

        # ============================================================
        # Debug information
        # ============================================================

        self.cloud_count += 1

        if self.cloud_count % 30 == 0:

            if total_points > 0:

                remove_ratio = (
                    float(removed_points)
                    / float(total_points)
                    * 100.0
                )

            else:

                remove_ratio = 0.0

            rospy.loginfo(
                "PointCloud: %d | "
                "Static: %d | "
                "Removed: %d | "
                "Dynamic ratio: %.1f%%",
                total_points,
                len(static_points),
                removed_points,
                remove_ratio
            )


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    try:

        node = DynamicPointCloudFilter()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass