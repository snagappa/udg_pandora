#!/usr/bin/env python

import roslib
roslib.load_manifest('learning_pandora')
import rospy

import numpy as np

from auv_msgs.msg import WorldWaypointReq


class test_waypoint():

    def __init__(self):
        print 'init !!!!!'
        self.pub_auv = rospy.Publisher(
            "/cola2_control/world_waypoint_req", WorldWaypointReq)

    def test(self):
        message = WorldWaypointReq()
        message.header.stamp = rospy.get_rostime()
        message.goal.requester = 'arnau'
        message.goal.id = 1
        message.goal.priority = 30
        message.altitude_mode =  False
        message.position.north =  0.4
        message.position.east =  -0.19
        message.position.depth = -0.27
        message.altitude =  0.0
        message.orientation.roll = 0.0
        message.orientation.pitch =  0.0
        message.orientation.yaw = 0.887
        message.disable_axis.x =  False
        message.disable_axis.y = False
        message.disable_axis.z = False
        message.disable_axis.roll = True
        message.disable_axis.pitch = True
        message.disable_axis.yaw = False
        message.position_tolerance.x= 0.5
        message.position_tolerance.y= 0.5
        message.position_tolerance.z= 0.5
        message.orientation_tolerance.roll= 0.0
        message.orientation_tolerance.pitch= 0.0
        message.orientation_tolerance.yaw= 0.5

        while not rospy.is_shutdown():
            print 'publish'
            message.header.stamp = rospy.get_rostime()
            self.pub_auv.publish(message)
            print 'Sleeping'
            rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('test_world_waypoint')

        print 'Calling object'
        result = test_waypoint()
        print 'Done'
        result.test()

    except rospy.ROSInterruptException:
        print "program interrupted before completion"
