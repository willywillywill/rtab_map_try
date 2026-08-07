#!/usr/bin/env python3

import math

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist


class TwistToAckermann:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/move_topic")
        self.output_topic = rospy.get_param("~output_topic", "/ackermann_cmd")
        self.frame_id = rospy.get_param("~frame_id", "base_link")

        self.wheelbase = rospy.get_param("~wheelbase", 0.235)

        self.min_speed = abs(rospy.get_param("~min_speed", 0.05))

        if self.wheelbase <= 0.0:
            rospy.logfatal("~wheelbase must be greater than 0")
            rospy.signal_shutdown("Invalid wheelbase")
            return

        self.publisher = rospy.Publisher(
            self.output_topic,
            AckermannDriveStamped,
            queue_size=10,
        )
        self.subscriber = rospy.Subscriber(
            self.input_topic,
            Twist,
            self.twist_callback,
            queue_size=10,
        )

        rospy.loginfo(
            "twist_to_ackermann: %s -> %s, wheelbase=%.3f m",
            self.input_topic,
            self.output_topic,
            self.wheelbase
        )

    def twist_callback(self, twist_msg):
        speed = twist_msg.linear.x
        yaw_rate = twist_msg.angular.z

        # Ackermann kinematics: yaw_rate = speed * tan(steering) / wheelbase
        if abs(speed) < self.min_speed:
            steering_angle = 0.0
        else:
            steering_angle = math.atan(self.wheelbase * yaw_rate / speed)

        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.header.frame_id = self.frame_id
        ackermann_msg.drive.speed = speed
        ackermann_msg.drive.steering_angle = steering_angle

        self.publisher.publish(ackermann_msg)


if __name__ == "__main__":
    rospy.init_node("twist_to_ackermann")
    TwistToAckermann()
    rospy.spin()