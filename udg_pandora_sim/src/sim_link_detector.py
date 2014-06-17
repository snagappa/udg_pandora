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
from visualization_msgs.msg import MarkerArray, Marker
from auv_msgs.msg import NavSts
import math 

class SimLinkDetector():
    def __init__(self, name):
        self.name = name
        self.nav = NavSts()
        
        # Config
        self.aris_tilt = math.radians(10)
        self.aris_elevation_angle = math.radians(14)
        self.aris_fov = math.radians(30)
        self.aris_window_start = 1.5
        self.aris_window_length = 3.5
        self.naviagtion_init = False
        self.chain_links = [[0, 0, 5.5], [0.5, 0, 5.5], [1.0, 0, 5.5], [1.5, 0, 5.5], [2.0, 0, 5.5]]
        
        # Create Publisher
        self.pub_marker = rospy.Publisher("/link_pose", MarkerArray)
        self.pub_aris_footprint = rospy.Publisher('/aris_foot_print', Marker)
        
        # Create Subscriber
        rospy.Subscriber("/cola2_navigation/nav_sts", NavSts,
                         self.update_nav_sts)

        # Enable Aris timer
        rospy.Timer(rospy.Duration(0.5), self.compute_link_detections)
        
        self.draw_chain_links()

    
    def update_nav_sts(self, nav_sts):
        self.nav = nav_sts
        self.naviagtion_init = True
        
    
    def compute_link_detections(self, event):
        result = self.compute_distances()
        self.draw_chain_links()
        
        if len(result) == 2:
            y1 = math.tan(self.aris_fov/2) * result[0]
            y2 = math.tan(self.aris_fov/2) * result[1]
            
            self.draw_marker(result[0], y1, self.nav.altitude, 0)
            self.draw_marker(result[0], -y1, self.nav.altitude, 1)
            self.draw_marker(result[1], y2, self.nav.altitude, 2)
            self.draw_marker(result[1], -y2, self.nav.altitude, 3)
            
    
    def compute_distances(self):
        if not self.naviagtion_init:
            rospy.loginfo("%s: Navigation not initialized", self.name)
            return False
       
        altitude = self.nav.altitude - 0.5
        
        angle_1 = (math.pi/2) - (self.aris_tilt + (self.aris_elevation_angle / 2) - self.nav.orientation.pitch) 
        angle_2 = (math.pi/2) - (self.aris_tilt - (self.aris_elevation_angle / 2) - self.nav.orientation.pitch)        
        # print "angles: ", math.degrees(angle_1), ", ", math.degrees(angle_2)
        # print "altitude: ", altitude
        
        if altitude < 0.1:
            rospy.loginfo("%s, Invalid altitude", self.name)
            return []

        hipo_1 = altitude / math.cos(angle_1)
        hipo_2 = altitude / math.cos(angle_2)
        # print "hipotenusa: ", hipo_1, ", ", hipo_2
            
        if hipo_1 > self.aris_window_start and hipo_2 > self.aris_window_start + self.aris_window_length:
            rospy.loginfo("%s, to high to see anything!", self.name)
            return []
            
        if hipo_1 <= self.aris_window_start and hipo_2 <= self.aris_window_start + self.aris_window_length:
            distance_1 = math.tan(angle_1)*altitude
            distance_2 = math.tan(angle_2)*altitude
            # print "distances: ", distance_1, ", ", distance_2
            print "Els dos per sota"
            return [distance_1, distance_2]
        else: # hipo_1 > self.aris_window_start and hipo_2 <= self.aris_window_start + self.aris_window_length:
            x1 = math.sin(angle_1)*self.aris_window_start
            y1 = -(math.cos(angle_1)*self.aris_window_start)
            x2 = math.sin(angle_2)*(self.aris_window_start + self.aris_window_length)
            y2 = -(math.cos(angle_2)*(self.aris_window_start + self.aris_window_length))
            # print 'Line 1: (', x1, ', ', y1, '), (', x2, ', ', y2, ')'
            a = (y2-y1)/(x2-x1)
            b = -x1*a + y1
            # print 'y = ', a, 'x + ', b
            print "Un per sota"
            # Compute intersection
            x = (-altitude - b)/a
            if hipo_1 > self.aris_window_start and hipo_2 <= self.aris_window_start + self.aris_window_length:
                distance_2 = math.tan(angle_2)*altitude            
                # print "Start windows parameter too short: ", x, ", ", distance_2
                return [x, distance_2]
            else:
                distance_1 = math.tan(angle_1)*altitude
                # print "Windows length parameter too short: ", distance_1, ", ", x
                return [distance_1, x]


    def draw_marker(self, x, y, z, marker_id):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "/girona500"

        marker.ns = "aris_foot_print_" + str(marker_id)
        marker.id = marker_id
        marker.type = 2 # SPHERE
        marker.action = 0 # Add/Modify an object
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.7
        marker.lifetime = rospy.Duration(0.5)
        marker.frame_locked = False
        self.pub_aris_footprint.publish(marker)
    
    def draw_chain_links(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "/world"

        marker.ns = "chain_links"
        for i in range(len(self.chain_links)):
            marker.id = 10 + i
            marker.type = 2 # SPHERE
            marker.action = 0 # Add/Modify an object
            marker.pose.position.x = self.chain_links[i][0]
            marker.pose.position.y = self.chain_links[i][1]
            marker.pose.position.z = self.chain_links[i][2]
            marker.scale.x = 0.7
            marker.scale.y = 0.35
            marker.scale.z = 0.1
            marker.color.r = 0.9
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.3
            marker.lifetime = rospy.Duration(10)
            marker.frame_locked = False
            self.pub_aris_footprint.publish(marker)
        
if __name__ == '__main__':
    try:
        rospy.init_node('sim_link_detector')
        SLD = SimLinkDetector(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
