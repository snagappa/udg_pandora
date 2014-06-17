#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on June 17 2014
@author: Narcis Palomeras
"""


# ROS imports
import roslib
roslib.load_manifest('cola2_perception_dev')
import rospy


class SimLinkDetector():
    def __init__(self, name):
        self.name = name

if __name__ == '__main__':
    try:
        rospy.init_node('sim_link_detector')
        SLD = SimLinkDetector(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
