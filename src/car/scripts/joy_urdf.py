#!/usr/bin/env python3

import math

import rospy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

class OdomToJointStates:
    def __init__(self):
        

        # =========================
        # 車輛尺寸
        # =========================
        self.wheelbase = rospy.get_param(
            "~wheelbase",
            0.235
        )

        self.track_width = rospy.get_param(
            "~track_width",
            0.13
        )

        self.wheel_radius = rospy.get_param(
            "~wheel_radius",
            0.0325
        )

        # 低於此速度時視為停止
        self.speed_epsilon = rospy.get_param(
            "~speed_epsilon",
            0.01
        )

        # 低於此角速度時視為直線
        self.yaw_rate_epsilon = rospy.get_param(
            "~yaw_rate_epsilon",
            0.001
        )

        # 停止時是否讓前輪回正
        self.center_steering_when_stopped = rospy.get_param(
            "~center_steering_when_stopped",
            False
        )

        # =========================
        # 車輪位置，單位 rad
        # =========================
        self.front_left_wheel_position = 0.0
        self.front_right_wheel_position = 0.0
        self.back_left_wheel_position = 0.0
        self.back_right_wheel_position = 0.0

        # 前輪轉向角，單位 rad
        self.front_left_steering = 0.0
        self.front_right_steering = 0.0

        self.last_update_time = None

        # =========================
        # ROS Publisher / Subscriber
        # =========================
        self.joint_pub = rospy.Publisher(
            "/joint_states",
            JointState,
            queue_size=10
        )

        self.odom_sub = rospy.Subscriber(
            "/odom",
            Odometry,
            self.odom_callback,
            queue_size=10
        )

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    @staticmethod
    def normalize_angle(angle):
        """將車輪旋轉位置限制在 -pi 到 pi。"""
        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def calculate_steering(self, linear_velocity, yaw_rate):
        """
        根據車體線速度和偏航角速度，
        計算左右前輪阿克曼轉向角。
        """
        if abs(linear_velocity) < self.speed_epsilon:
            if self.center_steering_when_stopped:
                self.front_left_steering = 0.0
                self.front_right_steering = 0.0

            # 停止時無法從 odom 推算轉向角
            return

        if abs(yaw_rate) < self.yaw_rate_epsilon:
            self.front_left_steering = 0.0
            self.front_right_steering = 0.0
            return

        # 車輛運動曲率 k = omega / v
        curvature = yaw_rate / linear_velocity

        left_denominator = (
            1.0 - curvature * self.track_width / 2.0
        )

        right_denominator = (
            1.0 + curvature * self.track_width / 2.0
        )

        denominator_epsilon = 0.001

        if abs(left_denominator) < denominator_epsilon:
            left_denominator = math.copysign(
                denominator_epsilon,
                left_denominator
                if left_denominator != 0.0
                else 1.0
            )

        if abs(right_denominator) < denominator_epsilon:
            right_denominator = math.copysign(
                denominator_epsilon,
                right_denominator
                if right_denominator != 0.0
                else 1.0
            )

        self.front_left_steering = math.atan(
            self.wheelbase
            * curvature
            / left_denominator
        )

        self.front_right_steering  = math.atan(
            self.wheelbase
            * curvature
            / right_denominator
        )

        print(self.front_left_steering, self.front_right_steering)

    def calculate_wheel_velocities(
        self,
        linear_velocity,
        yaw_rate
    ):
        """
        計算四個車輪的角速度，單位 rad/s。
        """
        if abs(linear_velocity) < self.speed_epsilon:
            return 0.0, 0.0, 0.0, 0.0

        if abs(yaw_rate) < self.yaw_rate_epsilon:
            wheel_angular_velocity = (
                linear_velocity / self.wheel_radius
            )

            return (
                wheel_angular_velocity,
                wheel_angular_velocity,
                wheel_angular_velocity,
                wheel_angular_velocity
            )

        curvature = yaw_rate / linear_velocity

        # 左右後輪的線速度
        back_left_linear_velocity = linear_velocity * (
            1.0 - curvature * self.track_width / 2.0
        )

        back_right_linear_velocity = linear_velocity * (
            1.0 + curvature * self.track_width / 2.0
        )

        # 前輪因為有轉向角，實際行走路徑較長
        left_cos = math.cos(
            self.front_left_steering
        )

        right_cos = math.cos(
            self.front_right_steering
        )

        if abs(left_cos) < 0.001:
            left_cos = math.copysign(
                0.001,
                left_cos if left_cos != 0.0 else 1.0
            )

        if abs(right_cos) < 0.001:
            right_cos = math.copysign(
                0.001,
                right_cos if right_cos != 0.0 else 1.0
            )

        front_left_linear_velocity = (
            back_left_linear_velocity / left_cos
        )

        front_right_linear_velocity = (
            back_right_linear_velocity / right_cos
        )

        # 線速度 m/s 轉為車輪角速度 rad/s
        front_left_angular_velocity = (
            front_left_linear_velocity / self.wheel_radius
        )

        front_right_angular_velocity = (
            front_right_linear_velocity / self.wheel_radius
        )

        back_left_angular_velocity = (
            back_left_linear_velocity / self.wheel_radius
        )

        back_right_angular_velocity = (
            back_right_linear_velocity / self.wheel_radius
        )

        return (
            front_left_angular_velocity,
            front_right_angular_velocity,
            back_left_angular_velocity,
            back_right_angular_velocity
        )

    def odom_callback(self, msg):
        now = msg.header.stamp

        # 若 odom 沒有時間戳，改用目前 ROS 時間
        if now == rospy.Time():
            now = rospy.Time.now()

        if self.last_update_time is None:
            self.last_update_time = now
            self.publish_joint_states(now)
            return

        dt = (now - self.last_update_time).to_sec()
        self.last_update_time = now

        # 防止時間倒退、重複或暫停太久
        if dt <= 0.0 or dt > 1.0:
            return

        linear_velocity = msg.twist.twist.linear.x
        yaw_rate = msg.twist.twist.angular.z

        # 計算左右前輪轉向角
        self.calculate_steering(
            linear_velocity,
            yaw_rate
        )

        (
            front_left_velocity,
            front_right_velocity,
            back_left_velocity,
            back_right_velocity
        ) = self.calculate_wheel_velocities(
            linear_velocity,
            yaw_rate
        )

        # 車輪角速度積分成位置
        self.front_left_wheel_position += (
            front_left_velocity * dt
        )

        self.front_right_wheel_position += (
            front_right_velocity * dt
        )

        self.back_left_wheel_position += (
            back_left_velocity * dt
        )

        self.back_right_wheel_position += (
            back_right_velocity * dt
        )

        # 防止數值無限增大
        self.front_left_wheel_position = (
            self.normalize_angle(
                self.front_left_wheel_position
            )
        )

        self.front_right_wheel_position = (
            self.normalize_angle(
                self.front_right_wheel_position
            )
        )

        self.back_left_wheel_position = (
            self.normalize_angle(
                self.back_left_wheel_position
            )
        )

        self.back_right_wheel_position = (
            self.normalize_angle(
                self.back_right_wheel_position
            )
        )

        self.publish_joint_states(now)

    def publish_joint_states(self, now):
        joint_state = JointState()

        joint_state.header.stamp = now

        # 必須與 URDF 裡的 joint name 完全相同
        joint_state.name = [
            "back_right_joint",
            "back_left_joint",
            "front_left_steer_joint",
            "front_left_wheel_joint",
            "front_right_steer_joint",
            "front_right_wheel_joint"
        ]

        joint_state.position = [
            self.back_right_wheel_position,
            self.back_left_wheel_position,
            self.front_left_steering,
            self.front_left_wheel_position,
            self.front_right_steering,
            self.front_right_wheel_position
        ]

        joint_state.velocity = []
        joint_state.effort = []

        self.joint_pub.publish(joint_state)


if __name__ == "__main__":
    rospy.init_node("odom_to_joint_states")

    node = OdomToJointStates()

    rospy.spin()