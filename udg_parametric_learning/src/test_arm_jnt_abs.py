#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_parametric_learning')
import rospy

from sensor_msgs.msg import Joy

rospy.init_node('current_estimation')

pub = rospy.Publisher('/cola2_control/joystick_arm_jnt_abs', Joy, queue_size=1)

message = Joy()
message.axes = [0.8, 0.0, 0.0, 0.0, 0.0]

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()
