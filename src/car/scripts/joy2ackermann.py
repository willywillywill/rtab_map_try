#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState
from ackermann_msgs.msg import AckermannDriveStamped

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

class joy2ackermann:
    
    def __init__(self):
        self.linear_axis = rospy.get_param("~linear_axis", 1)
        self.angular_axis = rospy.get_param("~angular_axis", 2)
        self.publish_rate = rospy.get_param("~publish_rate", 20)

        self.max_vx = rospy.get_param("~max_x", 1.0)
        self.max_wz = rospy.get_param("~max_z", 3.0)
        self.wheelbase = rospy.get_param("~wheelbase", 0.235)
        self.max_axis_x = 1.0
        self.max_axis_z = 1.0
        self.min_vx = 0.05

        self.vx = 0
        self.wz = 0
    
        self.joy_sub = rospy.Subscriber(
            "/joy",
            Joy,
            self.joy_callback
        )
        self.twist_sub = rospy.Subscriber(
            "/cmd_vel",
            Twist,
            self.twist_callback
        )

        self.ackermann_pub = rospy.Publisher(
            "/ackermann_cmd",
            AckermannDriveStamped,
            queue_size=10
        )

 

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            #print("cmd-> ", self.vx, self.wz)

            ackermann_msg = AckermannDriveStamped()
            ackermann_msg.header.stamp = rospy.Time.now()
            ackermann_msg.drive.speed = self.vx
            ackermann_msg.drive.steering_angle = self.wz
            self.ackermann_pub.publish(ackermann_msg)
            rate.sleep()
        

    def joy_callback(self, msg):
        linear_input = msg.axes[self.linear_axis]
        angular_input = msg.axes[self.angular_axis]

        vx = abs(self.max_vx/self.max_axis_x)*linear_input
        if (vx > self.max_vx): vx = self.max_vx

        wz = abs(self.max_wz/self.max_axis_z)*angular_input
        if (wz > self.max_wz): wz = self.max_wz

        self.vx = vx
        self.wz = wz


    def twist_callback(self, msg):
        speed = msg.linear.x
        yaw_rate = msg.angular.z

        if (abs(speed) < self.min_vx):
            angle = 0
        else:
            angle = math.atan(self.wheelbase * yaw_rate / speed)

            speed = clamp(abs(speed), 0.5, 1)
            
            if (speed < 0):
                speed *= -1
            
        print(speed, angle)
        self.vx = speed
        self.wz = angle

    

        
        


if __name__ == "__main__":

    rospy.init_node("joy2ackermann")
    j2a = joy2ackermann()
    j2a.run()
    