#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster

class Driver:
    def __init__(self, bot):
        self.bot = bot

        self.max_vx = rospy.get_param("~max_x", 1.0)
        self.max_wz = rospy.get_param("~max_z", 1.0)

        # 如果真車方向相反，再從 launch 調整
        self.invert_vx = rospy.get_param("~invert_vx", False)
        self.invert_wz = rospy.get_param("~invert_wz", False)

        self.subscriber = rospy.Subscriber(
            "/move_topic",
            Twist,
            self.cmd_callback,
            queue_size=1
        )

        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo("driver_node -> started")
        rospy.loginfo("driver_node -> max_vx: %.2f", self.max_vx)
        rospy.loginfo("driver_node -> max_wz: %.2f", self.max_wz)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def cmd_callback(self, msg):
        vx = self.clamp(
            msg.linear.x,
            -self.max_vx,
            self.max_vx
        )

        wz = self.clamp(
            msg.angular.z,
            -self.max_wz,
            self.max_wz
        )

        if self.invert_vx:
            vx = -vx

        if self.invert_wz:
            wz = -wz

        rospy.loginfo_throttle(
            0.2,
            "driver_node -> command vx=%.3f wz=%.3f",
            vx,
            wz
        )

        try:
            self.bot.set_car_motion(-vx, 0.0, wz)

        except Exception as error:
            rospy.logerr_throttle(
                1.0,
                "driver_node -> set_car_motion failed: %s",
                str(error)
            )

    def stop_robot(self):
        rospy.loginfo("driver_node -> stopping robot")

        try:
            self.bot.set_car_motion(0.0, 0.0, 0.0)

        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("move_subscriber")

    com_port = rospy.get_param(
        "~com_port",
        "/dev/ttyUSB1"
    )

    bot = None

    try:
        bot = Rosmaster(
            com=com_port,
        )

        driver = Driver(bot)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logfatal(
            "driver_node -> failed: %s",
            str(error)
        )

    finally:
        if bot is not None:
            try:
                bot.set_car_motion(0.0, 0.0, 0.0)
            except Exception:
                pass