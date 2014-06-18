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
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import tf
import numpy as np
import random

import math 

class SimLinkDetector():
    def __init__(self, name):
        self.name = name
        self.nav = NavSts()
        self.odometry = Odometry()
        
        # Config
        self.aris_tilt = math.radians(15)
        self.aris_elevation_angle = math.radians(14)
        self.aris_fov = math.radians(30)
        self.aris_window_start = 2.0
        self.aris_window_length = 3.5
        self.aris_hz = 5
        self.naviagtion_init = False
        self.chain_detections = MarkerArray()
        self.id = 0
        self.chain_links = [[0, 0, 5.5], 
                            [0.5, 0, 5.5], 
                            [1.0, 0, 5.5], 
                            [1.5, 0, 5.5], 
                            [2.0, 0, 5.5]]
        
        # Create Publisher
        self.pub_marker = rospy.Publisher("/link_pose", MarkerArray)
        self.pub_aris_footprint = rospy.Publisher('/aris_foot_print', Marker)
        
        # Create Subscriber
        rospy.Subscriber("/cola2_navigation/nav_sts", NavSts,
                         self.update_nav_sts)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry,
                         self.update_odometry)
        # Enable Aris timer
        rospy.Timer(rospy.Duration(1.0/self.aris_hz), 
                    self.compute_link_detections)
        
        

    
    def update_nav_sts(self, nav_sts):
        self.nav = nav_sts
        self.naviagtion_init = True
    
    
    def update_odometry(self, odom):
        self.odometry = odom

    
    def compute_link_detections(self, event):
        result = self.compute_distances()
        
        if len(result) == 2:
            y1 = math.tan(self.aris_fov/2) * result[0]
            y2 = math.tan(self.aris_fov/2) * result[1]
            
            footprint = [[result[0], y1, self.nav.altitude],
                         [result[1], y2, self.nav.altitude],
                         [result[1], -y2, self.nav.altitude],
                         [result[0], -y1, self.nav.altitude],
                         [result[0], y1, self.nav.altitude]]
            self.draw_footprint(footprint)
            
            [detections,
             link_pose] = self.compute_detetcted_links(footprint)
            self.draw_chain_links(detections)
            result = self.simulate_noisy_detection(detections, link_pose)
            if result != None:
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = "/girona500"
                marker.ns = "link_detection"
                marker.id = 100 + self.id
                self.id = self.id + 1
                marker.type = 0 # SPHERE
                marker.action = 0 # Add/Modify an object
                marker.pose.position.x = result[0]
                marker.pose.position.y = result[1]
                marker.pose.position.z = self.nav.altitude
                marker.scale.x = 0.2
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.7
                marker.lifetime = rospy.Duration(1)
                marker.frame_locked = False
                self.chain_detections.markers.append(marker)
        else:
            self.draw_chain_links([False for i in range(len(self.chain_links))])
        
        # Print current detections
        self.pub_marker.publish(self.chain_detections)
            
    def compute_distances(self):
        if not self.naviagtion_init:
            rospy.loginfo("%s: Navigation not initialized", self.name)
            return False
       
        altitude = self.nav.altitude - 0.5
        
        angle_1 = (math.pi/2)-(self.aris_tilt+(self.aris_elevation_angle/2)-\
                  self.nav.orientation.pitch) 
        angle_2 = (math.pi/2)-(self.aris_tilt-(self.aris_elevation_angle/2)-\
                  self.nav.orientation.pitch)        
        # print "angles: ", math.degrees(angle_1), ", ",math.degrees(angle_2)
        # print "altitude: ", altitude
        
        if altitude < 0.1:
            rospy.loginfo("%s, Invalid altitude", self.name)
            return []

        hipo_1 = altitude / math.cos(angle_1)
        hipo_2 = altitude / math.cos(angle_2)
        # print "hipotenusa: ", hipo_1, ", ", hipo_2
            
        if hipo_1 > self.aris_window_start and \
           hipo_2 > self.aris_window_start + self.aris_window_length:
            rospy.loginfo("%s, to high to see anything!", self.name)
            return []
            
        if hipo_1 <= self.aris_window_start and \
           hipo_2 <= self.aris_window_start + self.aris_window_length:
            distance_1 = math.tan(angle_1)*altitude
            distance_2 = math.tan(angle_2)*altitude
            # print "distances: ", distance_1, ", ", distance_2
            # print "Els dos per sota"
            return [distance_1, distance_2]
        else: 
            x1 = math.sin(angle_1)*self.aris_window_start
            y1 = -(math.cos(angle_1)*self.aris_window_start)
            x2 = math.sin(angle_2)*(self.aris_window_start +\
                                    self.aris_window_length)
            y2 = -(math.cos(angle_2)*(self.aris_window_start + \
                                      self.aris_window_length))
            # print 'Line 1: (', x1, ', ', y1, '), (', x2, ', ', y2, ')'
            a = (y2-y1)/(x2-x1)
            b = -x1*a + y1
            # print 'y = ', a, 'x + ', b
            # print "Un per sota"
            # Compute intersection
            x = (-altitude - b)/a
            if hipo_1 > self.aris_window_start and \
               hipo_2 <= self.aris_window_start + self.aris_window_length:
                distance_2 = math.tan(angle_2)*altitude            
                return [x, distance_2]
            else:
                distance_1 = math.tan(angle_1)*altitude
                return [distance_1, x]


    def draw_footprint(self, points):
        # print 'PRINT ARIS FOOTPRINT'
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "/girona500"

        marker.ns = "aris_foot_print"
        marker.id = 1
        marker.type = 4 # SPHERE
        marker.action = 0 # Add/Modify an object
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        for i in points:
            p = Point()
            p.x = i[0]
            p.y = i[1]
            p.z = i[2]
            color = ColorRGBA()
            color.r = 0.0
            color.g = 0.5
            color.b = 0.5
            color.a = 0.7
            marker.points.append(p)
            marker.colors.append(color)
            
        marker.lifetime = rospy.Duration(0.5)
        marker.frame_locked = False
        self.pub_aris_footprint.publish(marker)
    
    
    def draw_chain_links(self, detections):
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
            marker.color.r = 1.0
            if detections[i]:
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.g = 0.5
                marker.color.b = 0.5
            marker.color.a = 0.3
            marker.lifetime = rospy.Duration(1)
            marker.frame_locked = False
            self.pub_aris_footprint.publish(marker)
            
            
    def compute_detetcted_links(self, footprint):
        # Transform all waypoint from world frame to vehicle frame
        wTv = tf.transformations.quaternion_matrix(
                        [self.odometry.pose.pose.orientation.x,
                         self.odometry.pose.pose.orientation.y,
                         self.odometry.pose.pose.orientation.z,
                         self.odometry.pose.pose.orientation.w])
        wTv[0:3, 3] = [self.odometry.pose.pose.position.x,
                       self.odometry.pose.pose.position.y,
                       self.odometry.pose.pose.position.z]

        vTw = np.linalg.pinv(wTv)
        chain_links_wt_vehicle = list()
        for i in self.chain_links:
            Pw = np.array([0, 0, 0, 1.0])
            Pw[0] = i[0]
            Pw[1] = i[1]
            Pw[2] = i[2]
            Ps = np.dot(vTw, Pw)
            chain_link = Point()
            chain_link.x = Ps[0]
            chain_link.y = Ps[1]
            chain_link.z = Ps[2]
            # print 'Ps: ', chain_link  
            chain_links_wt_vehicle.append(chain_link)
      
      
        #  ARIS footprint
        #        _--1
        #    _--    |   
        #  0        |
        #  |        |
        #  3        |
        #   --__    |
        #       --__2
        #
        #        _--C
        #    _--    |   
        #  A--------B
        #  |        |
        #  D--------E
        #   --__    |
        #       --__F

        A = Point()
        A.x = footprint[0][0]
        A.y = footprint[0][1]
        A.z = footprint[0][2]
        B = Point()
        B.x = footprint[1][0]
        B.y = footprint[0][1]
        B.z = footprint[0][2]
        C = Point()
        C.x = footprint[1][0]
        C.y = footprint[1][1]
        C.z = footprint[1][2]
        E = Point()
        E.x = footprint[2][0]
        E.y = footprint[3][1]
        E.z = footprint[2][2]
        F = Point()
        F.x = footprint[2][0]
        F.y = footprint[2][1]
        F.z = footprint[2][2]
        D = Point()
        D.x = footprint[3][0]
        D.y = footprint[3][1]
        D.z = footprint[3][2]
        
        ret = []
        for i in chain_links_wt_vehicle:
            if __point_in_rectangle__(i, A, E) or \
               __point_in_triangle__(i, A, B, C) or \
               __point_in_triangle__(i, D, E, F):    
                ret.append(True)
            else:
                ret.append(False)
        return [ret, chain_links_wt_vehicle]
        
        
    def simulate_noisy_detection(self, detections, link_pose):
        r = (int)(random.random()*self.aris_hz)
        if r == 0: # Un image per second
            r = (int)(random.random()*len(detections))
            i = 0
            while i <  len(detections) and not detections[r]:
                i = i + 1
                r = (r + 1) % len(detections)
                
            if detections[r]:        
                return [link_pose[r].x + random.gauss(0.0, 0.05),
                        link_pose[r].y + random.gauss(0.0, 0.05)]
        return None                
            
def __same_side__(p1, p2, a, b):
    cp1 = np.cross(b-a, p1-a)
    cp2 = np.cross(b-a, p2-a)
    if np.dot(cp1, cp2) >= 0:
        return True
    else:
        return False
        
def __point_in_triangle__(p, a, b, c):
    p = np.array([p.x, p.y])
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    if __same_side__(p,a, b,c) and \
       __same_side__(p,b, a,c) and \
       __same_side__(p,c, a,b):
        return True
    else:
        return False
            
def __point_in_rectangle__(point, left_down_edge, right_top_edge):
    if point.x >= left_down_edge.x and \
       point.x <= right_top_edge.x and \
       point.y <= left_down_edge.y and \
       point.y >= right_top_edge.y:
        return True
    else:
        return False
           
        
if __name__ == '__main__':
    try:
        rospy.init_node('sim_link_detector')
        SLD = SimLinkDetector(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
