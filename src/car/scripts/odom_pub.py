#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf
import math
from Rosmaster_Lib import Rosmaster


def clamp_zero(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Odom:
    def __init__(self, bot):
        self.bot = bot

        self.pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.br = tf.TransformBroadcaster()

        self.x = 0.0
        self.y = 0.0

        self.yaw_offset = None
        self.last_yaw = 0.0

        self.last_time = rospy.Time.now()

        # 這幾個可以依照你的實際雜訊調整
        self.vx_deadband = rospy.get_param("~vx_deadband", 0.05)
        self.vy_deadband = rospy.get_param("~vy_deadband", 0.05)
        self.wz_deadband = rospy.get_param("~wz_deadband", 0.05)

    def run(self):
        rate = rospy.Rate(30)
        rospy.loginfo("odom_node -> start publishing odom")

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            if dt <= 0:
                dt = 0.0
            elif dt > 0.2:
                rospy.logwarn_throttle(
                    2.0,
                    "odom_node -> dt too large: %.3f, clamp to 0.05",
                    dt
                )
                dt = 0.05

            try:
                roll, pitch, imu_yaw = self.bot.get_imu_attitude_data(ToAngle=False)
                vx, vy, vz = self.bot.get_motion_data()
            except Exception as e:
                rospy.logwarn_throttle(
                    2.0,
                    "odom_node -> read Rosmaster failed: %s",
                    str(e)
                )
                rate.sleep()
                continue

            # 第一次啟動時，把目前 IMU yaw 當作 0 度
            if self.yaw_offset is None:
                self.yaw_offset = imu_yaw

            yaw = normalize_angle(imu_yaw - self.yaw_offset)

            # 阿克曼車不應該有側向速度
            vy = 0.0

            # deadband：小速度視為 0，避免靜止時 odom 飄
            vx = clamp_zero(vx, self.vx_deadband)
            vy = clamp_zero(vy, self.vy_deadband)
            vz = clamp_zero(vz, self.wz_deadband)

            # 如果車子幾乎沒動，固定 yaw，避免 IMU 漂移造成 RViz 裡旋轉
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

            q = tf.transformations.quaternion_from_euler(0, 0, yaw)

            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"

            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = Quaternion(*q)

            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.angular.z = vz

            self.pub.publish(odom)

            self.br.sendTransform(
                (self.x, self.y, 0.0),
                q,
                now,
                "base_link",
                "odom"
            )

            rospy.loginfo_throttle(
                1.0,
                "odom_node -> vx: %.3f, vy: %.3f, vz: %.3f, x: %.3f, y: %.3f, yaw: %.3f",
                vx, vy, vz, self.x, self.y, yaw
            )

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("odom_node")

    com_port = rospy.get_param("~com_port", "/dev/ttyUSB1")

    bot = Rosmaster(com=com_port)
    bot.create_receive_threading()

    odom = Odom(bot)
    odom.run()