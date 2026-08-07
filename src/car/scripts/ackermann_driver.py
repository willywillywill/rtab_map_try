#!/usr/bin/env python3

import math
import rospy

from ackermann_msgs.msg import AckermannDriveStamped
from Rosmaster_Lib import Rosmaster


class Driver:
    def __init__(self, bot):
        self.bot = bot

        # 車輛參數
        self.wheelbase = rospy.get_param("~wheelbase", 0.35)
        self.max_vx = rospy.get_param("~max_x", 1.0)
        self.max_wz = rospy.get_param("~max_z", 1.0)
        self.max_steering_angle = rospy.get_param(
            "~max_steering_angle",
            0.60
        )

        # 方向修正
        self.invert_vx = rospy.get_param("~invert_vx", False)
        self.invert_wz = rospy.get_param("~invert_wz", False)

        self.input_topic = rospy.get_param(
            "~input_topic",
            "/ackermann_cmd"
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            AckermannDriveStamped,
            self.cmd_callback,
            queue_size=1
        )

        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo("ackermann_driver -> started")
        rospy.loginfo("ackermann_driver -> topic: %s", self.input_topic)
        rospy.loginfo(
            "ackermann_driver -> wheelbase: %.3f m",
            self.wheelbase
        )
        rospy.loginfo(
            "ackermann_driver -> max_vx: %.3f m/s",
            self.max_vx
        )
        rospy.loginfo(
            "ackermann_driver -> max_wz: %.3f rad/s",
            self.max_wz
        )

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def cmd_callback(self, msg):
        # AckermannDriveStamped 中的速度與前輪轉向角
        vx = self.clamp(
            msg.drive.speed,
            -self.max_vx,
            self.max_vx
        )

        steering_angle = self.clamp(
            msg.drive.steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        # Ackermann 轉向角轉成車身角速度
        if abs(vx) > 0.001 and self.wheelbase > 0.0:
            wz = vx * math.tan(steering_angle) / self.wheelbase
        else:
            wz = 0.0

        wz = self.clamp(
            wz,
            -self.max_wz,
            self.max_wz
        )

        if self.invert_vx:
            vx = -vx

        if self.invert_wz:
            wz = -wz

        rospy.loginfo_throttle(
            0.2,
            "ackermann_driver -> vx=%.3f steering=%.3f wz=%.3f",
            vx,
            steering_angle,
            wz
        )

        try:
            self.bot.set_car_motion(vx, 0.0, wz)

        except Exception as error:
            rospy.logerr_throttle(
                1.0,
                "ackermann_driver -> set_car_motion failed: %s",
                str(error)
            )

    def stop_robot(self):
        rospy.loginfo("ackermann_driver -> stopping robot")

        try:
            self.bot.set_car_motion(0.0, 0.0, 0.0)
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("ackermann_driver")

    com_port = rospy.get_param(
        "~com_port",
        "/dev/ttyUSB1"
    )

    bot = None

    try:
        bot = Rosmaster(com=com_port)

        driver = Driver(bot)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logfatal(
            "ackermann_driver -> failed: %s",
            str(error)
        )

    finally:
        if bot is not None:
            try:
                bot.set_car_motion(0.0, 0.0, 0.0)
                del bot
            except Exception:
                pass