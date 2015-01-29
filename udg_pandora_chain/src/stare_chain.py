#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 28 2015
@author: narcis palomeras, tali
"""

# ROS imports
import roslib 
import rospy
import math

from auv_msgs.msg import WorldWaypointReq
from auv_msgs.msg import BodyForceReq
from auv_msgs.msg import BodyVelocityReq
from geometry_msgs.msg import PoseWithCovarianceStamped
from pose_ekf_slam.msg import Map
from cola2_control.msg import WorldWaypointReqAction, WorldWaypointReqGoal
from cola2_control.srv import StareLandmark, StareLandmarkResponse
from cola2_control.srv import SearchLandmark, SearchLandmarkResponse
from cola2_control.srv import PushWithAUV, PushWithAUVResponse
from std_srvs.srv import Empty, EmptyResponse
import PyKDL
import actionlib

from threading import Thread
from auv_msgs.msg import GoalDescriptor
from tf.transformations import euler_from_quaternion
from cola2_control.srv import Goto, GotoRequest
from auv_msgs.msg import NavSts

import tf

class StareChainController:
    
    def __init__(self, name):
        self.name = name
        self.map = Map()
        self.nav = NavSts()
        self.stare_landmark_init = False 
        self.offset_transformation = None
        self.keep_pose = False
        self.tolerance = 0.25
        self.push_init = False
        self.default_wrench = [45.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        
        self.tfbr = tf.TransformBroadcaster()
       
         # Create publisher
        self.pub_world_waypoint_req = rospy.Publisher(
                                        "/cola2_control/world_waypoint_req", 
                                        WorldWaypointReq,
                                        queue_size=2)

        # Create Subscriber
        rospy.Subscriber("/pose_ekf_slam/landmark_update/chain_position", 
                         PoseWithCovarianceStamped, 
                         self.update_chain_detector,
                         queue_size = 1)
                         
        rospy.Subscriber("/pose_ekf_slam/map", 
                         Map,
                         self.update_map,
                         queue_size = 1)

        rospy.Subscriber("/cola2_navigation/nav_sts", 
                         NavSts,
                         self.update_nav,
                         queue_size = 1)
        
        # Create Service         
        self.enable_stare_srv = rospy.Service('/cola2_control/enable_stare_chain', 
                                      StareLandmark, 
                                      self.stare_chain)
        
        self.disable_stare_srv = rospy.Service('/cola2_control/disable_stare_chain', 
                              Empty, 
                              self.disable)
           

        self.last_detection = rospy.Time.now()
        
    
    def update_chain_detector(self, data):
        self.last_detection = rospy.Time.now()
    
    
    def update_map(self, data):
        self.map = data

    def update_nav(self, data):
        self.nav = data

    
    def disable(self, req):
        self.stare_landmark_init = False
        return EmptyResponse()
       
 
    def stare_chain(self, req):
        # If another petition is being served return False
        if self.stare_landmark_init:
            rospy.logerr("Another petition is being served")
            return StareLandmarkResponse(False)
        
        # Otherwise check if the landmark exists
        i = 0
        found = False
        while i < len(self.map.landmark) and not found:
            if self.map.landmark[i].landmark_id == req.landmark_id:
                # Landmark found
                self.stare_landmark_init = True
                
                # Compute Transformation
                angle = euler_from_quaternion(
                                        [req.offset.orientation.x,
                                         req.offset.orientation.y,
                                         req.offset.orientation.z,
                                         req.offset.orientation.w])
                O = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
                t = PyKDL.Vector(req.offset.position.x,
                                  req.offset.position.y,
                                  req.offset.position.z)
                self.offset_transformation = PyKDL.Frame(O, t)
                self.keep_pose = req.keep_pose
                self.tolerance = req.tolerance
                t = Thread(target=self.stare_chain_thread, args=([i]))
                t.daemon = True
                t.start()
                if self.keep_pose:        
                    return StareLandmarkResponse(True)
                else:
                    t.join()
                    return StareLandmarkResponse(True)
            i = i + 1
            
        rospy.logerr("Invalid landmark ID")
        return StareLandmarkResponse(False)


    def stare_chain_thread(self, index):
        index = int(index)
        rate = rospy.Rate(10)
        while self.stare_landmark_init:
            # Compute Transformation
            angle = euler_from_quaternion(
                        [self.map.landmark[index].pose.pose.orientation.x,
                         self.map.landmark[index].pose.pose.orientation.y,
                         self.map.landmark[index].pose.pose.orientation.z,
                         self.map.landmark[index].pose.pose.orientation.w])
            O = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
            t = PyKDL.Vector(self.map.landmark[index].pose.pose.position.x,
                             self.map.landmark[index].pose.pose.position.y,
                             self.map.landmark[index].pose.pose.position.z)
            wTl = PyKDL.Frame(O, t)
            
            wTo = wTl * self.offset_transformation
            orientation = wTo.M.GetRPY()
            position = wTo.p
            wwr = WorldWaypointReq()
            wwr.header.stamp = rospy.Time().now()
            wwr.goal = GoalDescriptor(self.name, 
                                      1, 
                                      GoalDescriptor.PRIORITY_NORMAL)
                                      
            wwr.altitude_mode = False
            wwr.position.north = position[0]
            wwr.position.east = position[1]
            wwr.position.depth = position[2]
            wwr.altitude = 5.0
            wwr.orientation.roll = orientation[0]
            wwr.orientation.pitch = orientation[1]
            wwr.orientation.yaw = orientation[2]
            
            wwr.disable_axis.x = False
            wwr.disable_axis.y = False
            wwr.disable_axis.z = True
            wwr.disable_axis.roll =True
            wwr.disable_axis.pitch = True
            wwr.disable_axis.yaw = False
            
            self.pub_world_waypoint_req.publish(wwr)
            print 'publish:\n', wwr
            if not self.keep_pose:
                self.check_tolerance(wwr)
            rate.sleep()

                                  
    def check_tolerance(self, wwr):
        if ((abs(self.nav.position.north - wwr.position.north) < self.tolerance) and
            (abs(self.nav.position.east - wwr.position.east) < self.tolerance) and
            (abs(self.nav.orientation.yaw - wwr.orientation.yaw) < self.tolerance)):
                self.stare_landmark_init = False
        
        
if __name__ == '__main__':
    try:
        rospy.init_node('stare_chain')
        stare_landmark = StareChainController(rospy.get_name())
        rospy.spin() 
    except rospy.ROSInterruptException: 
        pass
