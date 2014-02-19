#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

from visualization_msgs.msg import MarkerArray
import tf


class ChainPlanner:
    def __init__(self, name):
        self.name = name


        # Create Subscriber Updates (z)
        rospy.Subscriber('/link_poses',
                         MarkerArray,
                         self.sonar_waypoint_update)

    def sonar_waypoint_update(self, data):
       pass        
   
if __name__ == '__main__':
    rospy.init_node('chain_planner')
    chain_follow = ChainPlanner(rospy.get_name())
    rospy.spin()
