#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 23/01/2014
@author: arnau carrera
"""
# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty, EmptyRequest
from cola2_control.srv import EFPose, EFPose

# Msgs imports
from cola2_control.srv import StareLandmark, StareLandmarkRequest

if __name__ == '__main__':
    try:
        name = 'landmark_stare_req'
        rospy.init_node(name)

        # Init Service Clients
        try:
            rospy.wait_for_service('/cola2_control/enable_stare_landmark', 20)
            stare_landmark_srv = rospy.ServiceProxy(
                                '/cola2_control/enable_stare_landmark', StareLandmark)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating stare landmark client')

        try:
            rospy.wait_for_service('/cola2_control/disable_stare_landmark', 20)
            disable_stare_landmark_srv = rospy.ServiceProxy(
                                '/cola2_control/disable_stare_landmark', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating disable_trajectory client')

        try:
            rospy.wait_for_service('/cola2_control/goto_landmark', 20)
            goto_landmark_srv = rospy.ServiceProxy(
                                '/cola2_control/goto_landmark', StareLandmark)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating goto landmark client')

        try:
            rospy.wait_for_service('/cola2_control/enable_push', 20)
            enable_push_srv = rospy.ServiceProxy(
                                '/cola2_control/enable_push', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating enable_push client')

        try:
            rospy.wait_for_service('/cola2_control/disable_push', 20)
            disable_push_srv = rospy.ServiceProxy(
                                '/cola2_control/disable_push', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating disable_push client')

        try:
            rospy.wait_for_service('/cola2_control/end_docking', 20)
            end_docking_srv = rospy.ServiceProxy(
                                '/cola2_control/end_docking', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating end_docking client')

        try:
            rospy.wait_for_service('/cola2_control/end_docking', 20)
            end_docking_srv = rospy.ServiceProxy(
                                '/cola2_control/end_docking', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating end_docking client')


        try:
            rospy.wait_for_service('/cola2_control/setPoseEF', 20)
            ef_pose_srv = rospy.ServiceProxy(
                '/cola2_control/setPoseEF', EFPose)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating setPoseEF client')

        req = StareLandmarkRequest()

        # Commented values are to swap between ARToolkit and Feauter-based detector
        req.landmark_id = "/pose_ekf_slam/landmark_update/panel_centre" #"/pose_ekf_slam/landmark_update/valve_1" # #
        # valve_dist_centre :
        # valve 0 : [-0.25, -0.125, 0.11]
        # valve 1 : [ 0.0, -0.125, 0.11]
        # valve 2 : [ 0.0, +0.125, 0.11]
        # valve 3 : [+0.25, -0.125, 0.11]]

        req.offset.position.x = 0.0 # 0.06
        req.offset.position.y = 0.125 # -0.4
        req.offset.position.z = 2.5

        quat = quaternion_from_euler(0.0, 1.57, 0.0)

        req.offset.orientation.x = quat[0]
        req.offset.orientation.y = quat[1]
        req.offset.orientation.z = quat[2]
        req.offset.orientation.w = quat[3]

        # Move to firts waypoint

        # Face Panel Far
        print 'Facing Panel ...'
        req.offset.position.z = 3.0
        ret = goto_landmark_srv(req)
        while not ret.attempted:
            rospy.sleep(5)
            ret = goto_landmark_srv(req)

        # Docking
        #print 'Wait ...'
        req.offset.position.z = 2.0
	req.tolerance = 0.25
        req.keep_pose = False
        #rospy.sleep(3)
        print 'Init docking ...'
        stare_landmark_srv(req)
        disable_stare_landmark_srv(EmptyRequest())
	print 'Push the panel ...'
        #enable_push_srv(EmptyRequest())
        #rospy.sleep(30)
        print 'Undocking ...'
        #disable_push_srv(EmptyRequest())
        end_docking_srv(EmptyRequest())
        print 'Docking finalized!'

    except rospy.ROSInterruptException:
        pass
