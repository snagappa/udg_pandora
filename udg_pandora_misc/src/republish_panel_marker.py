#!/usr/bin/env python
"""Created on 22 October 2013
author Arnau
"""
# ROS imports
import roslib
roslib.load_manifest('udg_pandora_misc')
import rospy
#import the map to read the data of the filter
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Pose
# from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

import numpy as np

import tf

class republish_panel_marker():
    """
    This node reads The map and publish the panel position as a marker array
    """

    def __init__(self, name):
        """
        This method load the configuration and initialize the publishers,
        subscribers and tf broadcaster and publishers
        """
        self.name = name
        #self.br = tf.TransformBroadcaster()
        self.tflistener = tf.TransformListener()
        # Create the publisher
        # we use a list to allow to have a variable number of valves
        self.panel_centre = Pose()

        #Comented because is not used
        self.pub_panel_marker = rospy.Publisher(
            "/pose_ekf_slam/panel", Marker)
        self.pub_valve0_marker = rospy.Publisher(
            "/pose_ekf_slam/valve0", Marker)
        self.pub_valve1_marker = rospy.Publisher(
            "/pose_ekf_slam/valve1", Marker)
        self.pub_valve2_marker = rospy.Publisher(
            "/pose_ekf_slam/valve2", Marker)
        self.pub_valve3_marker = rospy.Publisher(
            "/pose_ekf_slam/valve3", Marker)

        #subscrive to the Map where is the position of the center
        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.updatepanelpose,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve0",
                         PoseWithCovarianceStamped,
                         self.updatevalve0,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve1",
                         PoseWithCovarianceStamped,
                         self.updatevalve1,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve2",
                         PoseWithCovarianceStamped,
                         self.updatevalve2,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve3",
                         PoseWithCovarianceStamped,
                         self.updatevalve3,
                         queue_size = 1)

    def updatepanelpose(self, landmarkmap):
        """
        This method recive the data filtered from the ekf_map and publish the
        position for the valve
        """
        for mark in landmarkmap.landmark:
            #rospy.loginfo(' Lanmark ' +str(mark.landmark_id) + ' Config ' + str(self.landmark_id))
            if '/pose_ekf_slam/landmark_update/panel_centre' == mark.landmark_id:
                panel_centre = mark.pose.pose

                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp =  landmarkmap.header.stamp #rospy.Time.now()
                marker.ns = "panel"
                marker.id = 8
                marker.type = marker.CUBE # 1
                marker.action = marker.ADD
                marker.pose.position.x = panel_centre.position.x
                marker.pose.position.y = panel_centre.position.y
                marker.pose.position.z = panel_centre.position.z
                marker.pose.orientation.x = panel_centre.orientation.x
                marker.pose.orientation.y = panel_centre.orientation.y
                marker.pose.orientation.z = panel_centre.orientation.z
                marker.pose.orientation.w = panel_centre.orientation.w
                marker.scale.x = 0.8
                marker.scale.y = 0.6
                marker.scale.z = 0.1
                marker.color.a = 1.0 #Don't forget to set the alpha!
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                self.pub_panel_marker.publish(marker)

    def updatevalve0(self, valve_pose):
        """
        Publish a marker Cylinder for each valve
        """
        valve = valve_pose.pose.pose
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp =  valve_pose.header.stamp #rospy.Time.now()
        marker.ns = "valve0"
        marker.id = 10
        marker.type = marker.CYLINDER # 3
        marker.action = marker.ADD
        marker.pose.position.x = valve.position.x
        marker.pose.position.y = valve.position.y
        marker.pose.position.z = valve.position.z

        original_ori = tf.transformations.quaternion_matrix([
            valve.orientation.x,
            valve.orientation.y,
            valve.orientation.z,
            valve.orientation.w])
        transform_ori = tf.transformations.euler_matrix(
            1.57, 0.0, 1.57)
        res = np.dot(original_ori, transform_ori)
        quat = tf.transformations.quaternion_from_matrix(
                     res)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.15
        marker.color.a = 1.0 #Don't forget to set the alpha!
        marker.color.r = 0.6
        marker.color.g = 0.6
        marker.color.b = 0.6
        self.pub_valve0_marker.publish(marker)


    def updatevalve1(self, valve_pose):
        """
        Publish a marker Cylinder for each valve
        """
        valve = valve_pose.pose.pose
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp =  valve_pose.header.stamp #rospy.Time.now()
        marker.ns = "valve1"
        marker.id = 11
        marker.type = marker.CYLINDER # 3
        marker.action = marker.ADD
        marker.pose.position.x = valve.position.x
        marker.pose.position.y = valve.position.y
        marker.pose.position.z = valve.position.z

        original_ori = tf.transformations.quaternion_matrix([
            valve.orientation.x,
            valve.orientation.y,
            valve.orientation.z,
            valve.orientation.w])
        transform_ori = tf.transformations.euler_matrix(
            1.57, 0.0, 1.57)
        res = np.dot(original_ori, transform_ori)
        quat = tf.transformations.quaternion_from_matrix(
                     res)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.15
        marker.color.a = 1.0 #Don't forget to set the alpha!
        marker.color.r = 0.6
        marker.color.g = 0.6
        marker.color.b = 0.6
        self.pub_valve1_marker.publish(marker)


    def updatevalve2(self, valve_pose):
        """
        Publish a marker Cylinder for each valve
        """
        valve = valve_pose.pose.pose
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp =  valve_pose.header.stamp #rospy.Time.now()
        marker.ns = "valve2"
        marker.id = 12
        marker.type = marker.CYLINDER # 3
        marker.action = marker.ADD
        marker.pose.position.x = valve.position.x
        marker.pose.position.y = valve.position.y
        marker.pose.position.z = valve.position.z

        original_ori = tf.transformations.quaternion_matrix([
            valve.orientation.x,
            valve.orientation.y,
            valve.orientation.z,
            valve.orientation.w])
        transform_ori = tf.transformations.euler_matrix(
            1.57, 0.0, 1.57)
        res = np.dot(original_ori, transform_ori)
        quat = tf.transformations.quaternion_from_matrix(
                     res)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.15
        marker.color.a = 1.0 #Don't forget to set the alpha!
        marker.color.r = 0.6
        marker.color.g = 0.6
        marker.color.b = 0.6
        self.pub_valve2_marker.publish(marker)

    def updatevalve3(self, valve_pose):
        """
        Publish a marker Cylinder for each valve
        """
        valve = valve_pose.pose.pose
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp =  valve_pose.header.stamp #rospy.Time.now()
        marker.ns = "valve3"
        marker.id = 13
        marker.type = marker.CYLINDER # 3
        marker.action = marker.ADD
        marker.pose.position.x = valve.position.x
        marker.pose.position.y = valve.position.y
        marker.pose.position.z = valve.position.z

        original_ori = tf.transformations.quaternion_matrix([
            valve.orientation.x,
            valve.orientation.y,
            valve.orientation.z,
            valve.orientation.w])
        transform_ori = tf.transformations.euler_matrix(
            1.57, 0.0, 1.57)
        res = np.dot(original_ori, transform_ori)
        quat = tf.transformations.quaternion_from_matrix(
                     res)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.15
        marker.color.a = 1.0 #Don't forget to set the alpha!
        marker.color.r = 0.6
        marker.color.g = 0.6
        marker.color.b = 0.6
        self.pub_valve3_marker.publish(marker)

if __name__ == '__main__':
    try:
        rospy.init_node('republisher')
        republish = republish_panel_marker(rospy.get_name())
        #VALVETRACKER.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr('The  has stopped unexpectedly')
