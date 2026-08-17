#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState
from ackermann_msgs.msg import AckermannDriveStamped



class joy2ackermann:
    def __init__(self):
        self.linear_axis = rospy.get_param("~linear_axis", 1)
        self.angular_axis = rospy.get_param("~angular_axis", 2)

        self.max_vx = rospy.get_param("~max_x", 1.0)
        self.max_wz = rospy.get_param("~max_z", 3.0)
        self.max_axis_x = 1.0
        self.max_axis_z = 1.0
 
        self.joy_sub = rospy.Subscriber(
            "/joy",
            Joy,
            self.joy_callback
        )

        self.ackermann_pub = rospy.Publisher(
            "/joy2ackermann",
            AckermannDriveStamped,
            queue_size=10
        )
    
    def joy_callback(self, msg):
        linear_input = msg.axes[self.linear_axis]
        angular_input = msg.axes[self.angular_axis]

        vx = abs(self.max_vx/self.max_axis_x)*linear_input
        wz = abs(self.max_wz/self.max_axis_z)*angular_input

        print(f"Speed: {vx}, Angular Velocity: {wz}")

        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.drive.speed = vx
        ackermann_msg.drive.steering_angle = wz
        self.ackermann_pub.publish(ackermann_msg)


if __name__ == "__main__":

    rospy.init_node("joy2ackermann")
    _ = joy2ackermann()
    rospy.spin()
    