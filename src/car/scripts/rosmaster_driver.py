#!/usr/bin/env python3

import math

import rospy
import tf
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from Rosmaster_Lib import Rosmaster


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def clamp_zero(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value

class RosmasterDriver:
    def __init__(self, bot):
        self.bot = bot

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)

        self.x = 0.0
        self.y = 0.0
        self.yaw_offset = None
        self.last_time = rospy.Time.now()
        self.last_yaw = 0.0

        # 這幾個可以依照你的實際雜訊調整
        self.vx_deadband = rospy.get_param("~vx_deadband", 0.05)
        self.vy_deadband = rospy.get_param("~vy_deadband", 0.05)
        self.wz_deadband = rospy.get_param("~wz_deadband", 0.05)

        self.odom_pub = rospy.Publisher(
            "/odom",
            Odometry,
            queue_size=10,
        )
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.cmd_sub = rospy.Subscriber(
            "/ackermann_cmd",
            AckermannDriveStamped,
            self.command_callback,
            queue_size=10,
        )
        self.bot.set_car_motion(0.0, 0.0, 0.0)

        rospy.on_shutdown(self.shutdown)

    def command_callback(self, msg):
        speed = msg.drive.speed
        steering_angle = msg.drive.steering_angle

        try:
            self.bot.set_car_motion(speed, 0.0, steering_angle)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "rosmaster_driver -> motor command failed: %s",
                str(exc),
            )

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        rospy.loginfo("rosmaster_driver -> started")

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(0.0, (now - self.last_time).to_sec())
            self.last_time = now

            try:
                _, _, imu_yaw = self.bot.get_imu_attitude_data(ToAngle=False)
                vx, vy, vz = self.bot.get_motion_data()
            except Exception as exc:
                rospy.logerr("rosmaster_driver -> serial read failed: %s", str(exc))
                rospy.signal_shutdown("Rosmaster serial connection lost")
                break

            if self.yaw_offset is None:
                self.yaw_offset = imu_yaw

            yaw = normalize_angle(imu_yaw - self.yaw_offset)

            vy = 0.0

            vx = clamp_zero(vx, self.vx_deadband)
            vy = clamp_zero(vy, self.vy_deadband)
            vz = clamp_zero(vz, self.wz_deadband)

            if abs(vx) < self.vx_deadband and abs(vz) < self.wz_deadband:
                vx = 0.0
                vy = 0.0
                vz = 0.0
                yaw = self.last_yaw
            else:
                self.last_yaw = yaw

            vx_world = vx * math.cos(yaw) - vy * math.sin(yaw)
            vy_world = vx * math.sin(yaw) + vy * math.cos(yaw)
            self.x += vx_world * dt
            self.y += vy_world * dt

            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame

            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = quaternion[0]
            odom.pose.pose.orientation.y = quaternion[1]
            odom.pose.pose.orientation.z = quaternion[2]
            odom.pose.pose.orientation.w = quaternion[3]

            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = vz

            self.odom_pub.publish(odom)
            self.tf_broadcaster.sendTransform(
                (self.x, self.y, 0.0),
                quaternion,
                now,
                self.base_frame,
                self.odom_frame,
            )

            rospy.loginfo_throttle(
                1.0,
                "rosmaster_driver -> vx=%.3f vy=%.3f wz=%.3f x=%.3f y=%.3f yaw=%.3f",
                vx,
                vy,
                vz,
                self.x,
                self.y,
                yaw,
            )
            rate.sleep()

    def shutdown(self):
        rospy.loginfo("rosmaster_driver -> stopping car")
        try:
            self.bot.set_car_motion(0.0, 0.0, 0.0)
        except Exception as exc:
            rospy.logwarn("rosmaster_driver -> stop command failed: %s", str(exc))


if __name__ == "__main__":
    rospy.init_node("rosmaster_driver")

    com_port = rospy.get_param("~com_port", "/dev/ttyUSB0")
    bot = None

    try:
        bot = Rosmaster(com=com_port)
        bot.create_receive_threading()

        driver = RosmasterDriver(bot)
        driver.run()
    except Exception as exc:
        rospy.logfatal("rosmaster_driver -> startup failed: %s", str(exc))
        if bot is not None:
            try:
                bot.set_car_motion(0.0, 0.0, 0.0)
            except Exception:
                pass