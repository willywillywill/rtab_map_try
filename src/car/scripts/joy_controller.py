#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState


class JoyController:
    def __init__(self):
        # 車輛控制參數
        self.max_x = rospy.get_param("~max_x", 1.8)

        # 建議先從 1.0 測試，3.0 可能轉向過於劇烈
        self.max_z = rospy.get_param("~max_z", 1.0)

        self.deadzone = rospy.get_param("~deadzone", 0.05)

        # 搖桿 axis
        self.linear_axis = rospy.get_param("~linear_axis", 1)
        self.angular_axis = rospy.get_param("~angular_axis", 2)
        self.invert_linear = rospy.get_param("~invert_linear", True)
        self.invert_angular = rospy.get_param("~invert_angular", False)

        # URDF 車輛尺寸
        self.wheelbase = rospy.get_param("~wheelbase", 0.235)
        self.track_width = rospy.get_param("~track_width", 0.13)
        self.wheel_radius = rospy.get_param("~wheel_radius", 0.0325)

        # URDF 前輪最大轉向角
        self.max_steering = rospy.get_param("~max_steering", 3)

        # 搖桿逾時保護
        self.joy_timeout = rospy.get_param("~joy_timeout", 0.5)

        self.cmd_pub = rospy.Publisher(
            "/move_topic",
            Twist,
            queue_size=1
        )

        self.joint_pub = rospy.Publisher(
            "/joint_states",
            JointState,
            queue_size=10
        )

        self.joy_sub = rospy.Subscriber(
            "/joy",
            Joy,
            self.joy_callback,
            queue_size=1
        )

        # 當前狀態
        self.linear_input = 0.0
        self.angular_input = 0.0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.front_left_steering = 0.0
        self.front_right_steering = 0.0

        self.front_left_wheel_position = 0.0
        self.front_right_wheel_position = 0.0
        self.back_left_wheel_position = 0.0
        self.back_right_wheel_position = 0.0

        self.last_joy_time = rospy.Time.now()
        self.last_update_time = rospy.Time.now()

        # 固定 30 Hz 發布 joint_states
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / 30.0),
            self.update
        )

        rospy.loginfo(
            "joy_controller -> started, linear_axis=%d angular_axis=%d",
            self.linear_axis,
            self.angular_axis
        )

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_axis(self, msg, index):
        if index < 0 or index >= len(msg.axes):
            rospy.logwarn_throttle(
                2.0,
                "joy_controller -> axis %d does not exist, axes size=%d",
                index,
                len(msg.axes)
            )
            return 0.0

        return msg.axes[index]

    def calculate_ackermann_steering(self, center_angle):
        """
        將中央轉向角轉換成阿克曼左右輪角度。
        左轉為正，右轉為負。
        """
        if abs(center_angle) < 0.0001:
            return 0.0, 0.0

        direction = 1.0 if center_angle > 0.0 else -1.0
        center_angle_abs = abs(center_angle)

        turning_radius = self.wheelbase / math.tan(center_angle_abs)

        inner_denominator = max(
            0.001,
            turning_radius - self.track_width / 2.0
        )

        outer_denominator = (
            turning_radius + self.track_width / 2.0
        )

        inner_angle = math.atan(
            self.wheelbase / inner_denominator
        )

        outer_angle = math.atan(
            self.wheelbase / outer_denominator
        )

        inner_angle = self.clamp(
            inner_angle,
            0.0,
            self.max_steering
        )

        outer_angle = self.clamp(
            outer_angle,
            0.0,
            self.max_steering
        )

        if direction > 0.0:
            # 左轉：左輪是內輪
            left_angle = inner_angle
            right_angle = outer_angle
        else:
            # 右轉：右輪是內輪
            left_angle = -outer_angle
            right_angle = -inner_angle

        return left_angle, right_angle

    def joy_callback(self, msg):
        linear = self.get_axis(msg, self.linear_axis)
        angular = self.get_axis(msg, self.angular_axis)

        if self.invert_linear:
            linear = -linear

        if self.invert_angular:
            angular = -angular

        linear = self.apply_deadzone(linear)
        angular = self.apply_deadzone(angular)

        self.linear_input = self.clamp(
            linear,
            -1.0,
            1.0
        )

        self.angular_input = self.clamp(
            angular,
            -1.0,
            1.0
        )

        self.linear_velocity = (
            self.linear_input * self.max_x
        )

        # 傳給 Rosmaster 的旋轉控制值
        self.angular_velocity = (
            self.angular_input * self.max_z
        )

        # RViz 中顯示的中央轉向角
        center_steering = (
            self.angular_input * self.max_steering
        )

        (
            self.front_left_steering,
            self.front_right_steering
        ) = self.calculate_ackermann_steering(
            center_steering
        )

        self.last_joy_time = rospy.Time.now()

        self.publish_twist()

    def publish_twist(self):
        command = Twist()

        command.linear.x = self.linear_velocity
        command.linear.y = 0.0
        command.linear.z = 0.0

        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = self.angular_velocity

        self.cmd_pub.publish(command)

    def publish_joint_states(self, now):
        joint_state = JointState()

        joint_state.header.stamp = now

        # 名稱必須和 URDF 完全一致
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

    def update_wheel_positions(self, dt):
        if self.wheel_radius <= 0.0:
            return

        wheel_angular_velocity = (
            self.linear_velocity / self.wheel_radius
        )

        wheel_delta = wheel_angular_velocity * dt

        self.back_left_wheel_position += wheel_delta
        self.back_right_wheel_position += wheel_delta
        self.front_left_wheel_position += wheel_delta
        self.front_right_wheel_position += wheel_delta

        # 避免角度數值無限增大
        self.back_left_wheel_position = math.fmod(
            self.back_left_wheel_position,
            2.0 * math.pi
        )

        self.back_right_wheel_position = math.fmod(
            self.back_right_wheel_position,
            2.0 * math.pi
        )

        self.front_left_wheel_position = math.fmod(
            self.front_left_wheel_position,
            2.0 * math.pi
        )

        self.front_right_wheel_position = math.fmod(
            self.front_right_wheel_position,
            2.0 * math.pi
        )

    def stop_robot(self):
        self.linear_input = 0.0
        self.angular_input = 0.0

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # 搖桿逾時後前輪回正
        self.front_left_steering = 0.0
        self.front_right_steering = 0.0

        self.publish_twist()

    def update(self, event):
        now = rospy.Time.now()
        dt = (now - self.last_update_time).to_sec()
        self.last_update_time = now

        if dt < 0.0 or dt > 0.2:
            dt = 1.0 / 30.0

        joy_age = (now - self.last_joy_time).to_sec()

        if joy_age > self.joy_timeout:
            if (
                self.linear_velocity != 0.0
                or self.angular_velocity != 0.0
            ):
                rospy.logwarn(
                    "joy_controller -> joystick timeout, stop"
                )

            self.stop_robot()

        self.update_wheel_positions(dt)
        self.publish_joint_states(now)

        rospy.loginfo_throttle(
            1.0,
            "joy_controller -> vx=%.2f wz=%.2f "
            "left=%.2f right=%.2f",
            self.linear_velocity,
            self.angular_velocity,
            self.front_left_steering,
            self.front_right_steering
        )


if __name__ == "__main__":
    rospy.init_node("joy_controller")

    try:
        JoyController()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass