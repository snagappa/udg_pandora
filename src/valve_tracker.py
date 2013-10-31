#!/usr/bin/env python
"""Created on 22 October 2013
author Arnau
"""
# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
#use to load the configuration function
import cola2_ros_lib
#import the map to read the data of the filter
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from std_msgs.msg import Float64

import numpy as np

import tf

import threading


class valveTracker():
    """
    This class Track the different pose of the valves. The information of the
    different cameras is combined and filtered to obtain a regular and robust
    pose of each valve.
    """

    def __init__(self, name):
        """
        This method load the configuration and initialize the publishers,
        subscribers and tf broadcaster and publishers
        """
        self.name = name
        self.getconfig()
        # Created a broadcaster for the TK of each joint of the arm
        #self.br = tf.TransformBroadcaster()
        self.tflistener = tf.TransformListener()
        # Create the publisher
        # we use a list to allow to have a variable number of valves
        self.valve_publishers = []
        self.valve_poses = []
        self.valve_ori_pub = []
        self.valve_ori_cov = []
        self.last_update_tf = []
        time = rospy.Time.now()
        self.last_update_ee_tf = time
        for i in xrange(self.num_valves):
            pub_name = '/valve_tracker/valve'+str(i)
            self.valve_publishers.append(rospy.Publisher(
                pub_name, PoseWithCovarianceStamped))
            self.valve_poses.append([0, 0, 0])
            pub_name_ori = '/valve_tracker/valve_ori'+str(i)
            self.valve_ori_pub.append(rospy.Publisher(
                pub_name_ori, Float64))
            pub_name_cov = '/valve_tracker/valve_ori_cov'+str(i)
            self.valve_ori_cov.append(rospy.Publisher(
                pub_name_cov, Float64))
            self.last_update_tf.append(time)
        #subscrive to the Map where is the position of the center
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updatepanelpose)
        rospy.Subscriber("/pose_ekf_slam/landmark_update/panel_centre",
                         PoseWithCovarianceStamped, self.updatecovariance)

        self.lock = threading.Lock()
        self.lock_error = threading.Lock()
        self.panel_centre = Pose()

        #Linear Kalman Filter
        # oritentation parameters initial guess
        self.kf_valves_ori = np.zeros(self.num_valves)
        self.kf_q_error = np.ones(self.num_valves)*1e-5
        self.kf_p = np.ones(self.num_valves)

        # State transition matrix
        self.kf_a = np.ones(self.num_valves)
        # Control matrix
        self.kf_b = np.ones(self.num_valves)
        # The control vector will be 0

        #observations
        self.kf_r_error = np.ones(self.num_valves)*1e-5
        #observation matrix
        self.kf_h = np.ones(self.num_valves)

        #innovation
        self.kf_innov = np.zeros(self.num_valves)
        self.kf_innov_cov = np.zeros(self.num_valves)

        #predictions
        self.kf_p_hat = np.ones(self.num_valves)
        self.kf_valves_ori_hat = np.zeros(self.num_valves)

    def getconfig(self):
        """
        This method load the configuration and initialize the publishers,
        subscribers and tf broadcaster and publishers
        """
        param_dict = {'period': '/valve_tracker/period',
                      'num_valves': '/valve_tracker/number_of_valves',
                      'name_valve_pose_ee': '/valve_tracker/name_valve_pose_ee',
                      'name_valve_ori_ee': '/valve_tracker/name_valve_ori_ee',
                      'landmark_id': '/valve_tracker/landmark_id',
                      'valve_dist_centre': '/valve_tracker/valve_dist_centre',
                      'valve_dist_centre_ee': '/valve_tracker/valve_dist_centre_ee',
                      'valve_id': '/valve_tracker/valve_id'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def updatebumbleebetf(self):
        """
        This method check if there is a tf with the position of the valve pose
        provided using the bumblebee
        """
        for i in xrange(self.num_valves):
            try:
                # trans, rot = self.tflistener.lookupTransform(
                #     'world', ('valve'+str(i)),
                #     self.tflistener.getLatestCommonTime(
                #         'world', 'valve'+str(i)))
                trans, rot = self.tflistener.lookupTransform(
                    '/panel_centre', ('valve'+str(i)),
                    self.tflistener.getLatestCommonTime(
                        '/panel_centre', 'valve'+str(i)))
                #yaw
                #reading measurement
                time = self.tflistener.getLatestCommonTime(
                    '/panel_centre', 'valve'+str(i))
                if self.last_update_tf[i] < time:
                    self.last_update_tf[i] = time
                    euler = tf.transformations.euler_from_quaternion(
                        rot)
                    measurement = tf.transformations.euler_from_quaternion(
                        rot)[2]
                    #rospy.loginfo('Measurement valve ' + str(i) + ' : ' + str(measurement))
                    #rospy.loginfo('Euler valve ' + str(i) + ' : ' + str(euler))
                    #rospy.loginfo('Time ' + str(i) +
                    #              str(self.tflistener.getLatestCommonTime(
                    #                  'world', 'valve'+str(i))))
                    #--------------Observation step----------------
                    #innovation = measurement_vector -
                    #             self.H*predicted_state_estimate
                    self.kf_innov[i] = (measurement -
                                        self.kf_h[i] * self.kf_valves_ori_hat[i])
                    #innovation_covariance = self.H*predicted_prob_estimate*
                    #                        numpy.transpose(self.H) + self.R
                    # if i == 2 :
                    #     rospy.loginfo('***********************************************')
                    #     rospy.loginfo('Innovation 2        = ' + str(self.kf_innov[i]) )
                    #     rospy.loginfo('Predicted value 2   = ' + str(self.kf_valves_ori_hat[i]))
                    #     rospy.loginfo('Measurement valve 2 = ' + str(measurement))
                    #     rospy.loginfo('***********************************************')
                    self.lock_error.acquire()
                    try:
                        self.kf_innov_cov[i] = (self.kf_h[i] * self.kf_p_hat[i] *
                                                self.kf_h[i] + self.kf_r_error[i])
                    finally:
                        self.lock_error.release()
                else:
                    self.kf_innov[i] = 0.0
                    self.kf_innov_cov[i] = 0.0
            except tf.Exception:
                self.kf_innov[i] = 0.0
                self.kf_innov_cov[i] = 0.0
                # rospy.logerr(
                #     'Error reading the Transformation from world to EE')
        #rospy.loginfo('***********************************************')

    def updatehandcameraposetf(self):
        """
        This method check if there is a tf with the position of the valve pose
        provided using the camera in the end-effector
        """
        try:
            trans, rot = self.tflistener.lookupTransform(
                'world', self.name_valve_pose_ee,
                self.tflistener.getLatestCommonTime(
                    'world', self.name_valve_pose_ee))
        except tf.Exception:
            pass
            # rospy.logerr(
            #     'Error reading the Tranformation from world to EE')

    def updatehandcameraoritf(self):
        """
        This method check if there is a tf with the position and orientation of
        the valve pose provided using the bumblebee
        """
        try:
            trans, rot = self.tflistener.lookupTransform(
                'base_arm', self.name_valve_ori_ee,
                self.tflistener.getLatestCommonTime(
                    'base_arm', self.name_valve_ori_ee))
            #yaw
            #reading measurement
            time = self.tflistener.getLatestCommonTime(
                '/base_arm', self.name_valve_ori_ee)
            if self.last_update_ee_tf < time:
                i = self.valve_id
                self.last_update_ee_tf = time
                measurement = tf.transformations.euler_from_quaternion(
                    rot)[2]
                #--------------Observation step----------------
                #innovation = measurement_vector -
                #             self.H*predicted_state_estimate
                self.kf_innov[i] = (measurement -
                                    self.kf_h[i] * self.kf_valves_ori_hat[i])
                #innovation_covariance = self.H*predicted_prob_estimate*
                #                        numpy.transpose(self.H) + self.R
                self.lock_error.acquire()
                try:
                    # TODO: Cacula la invertesa.
                    self.kf_innov_cov[i] = (self.kf_h[i] * self.kf_p_hat[i] *
                                            self.kf_h[i] + self.kf_r_error[i])
                finally:
                    self.lock_error.release()
            else:
                self.kf_innov[i] = 0.0
                self.kf_innov_cov[i] = 0.0
        except tf.Exception:
            pass
            # rospy.logerr(
            #     'Error reading the Tranformation from world to EE')

    def updatepanelpose(self, landmarkmap):
        """
        This method recive the data filtered from the ekf_map and publish the
        position for the valve
        """
        self.lock.acquire()
        try:
            for mark in landmarkmap.landmark:
                #rospy.loginfo(' Lanmark ' +str(mark.landmark_id) + ' Config ' + str(self.landmark_id))
                if self.landmark_id == mark.landmark_id:
                    self.panel_centre = mark.pose.pose
                    #Create the Transformation matrix
                    trans_mat = tf.transformations.quaternion_matrix(
                        [self.panel_centre.orientation.x,
                         self.panel_centre.orientation.y,
                         self.panel_centre.orientation.z,
                         self.panel_centre.orientation.w])
                    trans_mat[0, 3] = self.panel_centre.position.x
                    trans_mat[1, 3] = self.panel_centre.position.y
                    trans_mat[2, 3] = self.panel_centre.position.z

                    #invert the Matrix
                    # inv_mat = np.zeros([4, 4])
                    # inv_mat[3, 3] = 1.0
                    # inv_mat[0:3, 0:3] = np.transpose(trans_mat[0:3, 0:3])
                    # inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                    #                          trans_mat[0:3, 3])

                    # rospy.loginfo('Panel Centre ' +
                    #               str(self.panel_centre.position.x) +
                    #               str(self.panel_centre.position.y) +
                    #               str(self.panel_centre.position.z) )
                    for i in xrange(self.num_valves):
                        valve_pose = np.ones(4)
                        valve_pose[0:3] = self.valve_dist_centre[i]
                        self.valve_poses[i] = np.dot(trans_mat, valve_pose)
                        v_pose = PoseWithCovarianceStamped()
                        v_pose.pose.pose.position.x = self.valve_poses[i][0]
                        v_pose.pose.pose.position.y = self.valve_poses[i][1]
                        v_pose.pose.pose.position.z = self.valve_poses[i][2]
                        v_pose.pose.pose.orientation = self.panel_centre.orientation
                        v_pose.pose.covariance = mark.pose.covariance
                        v_pose.header.stamp = rospy.Time.now()
                        self.valve_publishers[i].publish(v_pose)
                        # rospy.loginfo('Valve ' + str(i) + ' : '
                        #               + str(self.valve_poses[i]))
        finally:
            self.lock.release()

    def updatecovariance(self, msg):
        """
        This method recive the data published by the visual detector and obtain
        the covariance of the position to use in the KF
        """
        self.lock_error.acquire()
        try:
            self.kf_r_error = np.ones(self.num_valves) * msg.pose.covariance[0]
        finally:
            self.lock_error.release()

    def predictpose(self):
        """
        This method recive the data filtered from the ekf_map and publish the
        position for the valve
        """
        #We supose the valve is not moving so the control is 0
        #predicted_state_estimate = self.A * self.current_state_estimate
        #                           + self.B * control_vector
        self.kf_valves_ori_hat = (self.kf_a * self.kf_valves_ori +
                                  self.kf_b * 0.0)
        # predicted_prob_estimate = (self.A * self.current_prob_estimate) *
        #                            numpy.transpose(self.A) + self.Q
        self.kf_p_hat = (self.kf_a * self.kf_p)*self.kf_a + self.kf_q_error

    def updatekf(self):
        """
        update the value of the kalman filter value for each filter
        """
        #-----------------------------Update step-------------------------------
        #kalman_gain = predicted_prob_estimate * numpy.transpose(self.H)
        #              * numpy.linalg.inv(innovation_covariance)
        kalman_gain = np.zeros(self.num_valves)
        for i in xrange(self.num_valves):
            if self.kf_innov_cov[i] == 0.0:
                self.kf_valves_ori[i] = self.kf_valves_ori_hat[i]
                self.kf_p[i] = self.kf_p_hat[i]
            else:
                kalman_gain[i] = (self.kf_p_hat[i] * self.kf_h[i] *
                                  (1/self.kf_innov_cov[i]))
                # if i == 2 :
                #     rospy.loginfo('x_hat ' + str(i) + ' ' + str(self.kf_valves_ori_hat[i]))
                #     rospy.loginfo('innov_cov ' + str(i) + ' ' + str(1/self.kf_innov_cov[i]))
                self.kf_valves_ori[i] = (self.kf_valves_ori_hat[i] +
                                         kalman_gain[i] * self.kf_innov[i])
                self.kf_p[i] = (1-kalman_gain[i]*self.kf_h[i])*self.kf_p_hat[i]
            # self.valve_ori_pub[i].publish(self.kf_valves_ori[i])
            # self.valve_ori_cov[i].publish(self.kf_p[i])
        #rospy.loginfo('********************************************')

    def updatekfhand(self):
        """
        update the value of the kalman filter value of the end_effector
        """
        #-----------------------------Update step-------------------------------
        #kalman_gain = predicted_prob_estimate * numpy.transpose(self.H)
        #              * numpy.linalg.inv(innovation_covariance)
        kalman_gain = 0.0
        i = self.valve_id
        if self.kf_innov_cov[i] == 0.0:
            self.kf_valves_ori[i] = self.kf_valves_ori_hat[i]
            self.kf_p[i] = self.kf_p_hat[i]
        else:
            kalman_gain = (self.kf_p_hat[i] * self.kf_h[i] *
                           (1/self.kf_innov_cov[i]))
            self.kf_valves_ori[i] = (self.kf_valves_ori_hat[i] +
                                     kalman_gain * self.kf_innov[i])
            self.kf_p[i] = (1-kalman_gain*self.kf_h[i])*self.kf_p_hat[i]

    def publish(self):
        """
        Publish the data updated in the kalman filter
        """
        for i in xrange(self.num_valves):
            self.valve_ori_pub[i].publish(self.kf_valves_ori[i])
            self.valve_ori_cov[i].publish(self.kf_p[i])

    def run(self):
        """
        This is the main loop where the prediction of the filter is done and the
        pose and orientation of each valve is published with a regular frequency
        """
        while not rospy.is_shutdown():
            self.predictpose()
            self.updatebumbleebetf()
            self.updatekf()
            self.updatehandcameraoritf()
            self.updatekfhand()
            self.publish()
            #Publish the Pose
            rospy.sleep(self.period)

if __name__ == '__main__':
    try:
        import subprocess
        # Load ROS parameters
        config_file_list = roslib.packages.find_resource("udg_pandora",
            "valve_tracker.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            print "Could not locate visual_detector.yaml, using defaults"
            set_default_parameters()
        rospy.init_node('valve_tracker')
        VALVETRACKER = valveTracker(rospy.get_name())
        VALVETRACKER.run()
    except rospy.ROSInterruptException:
        rospy.logerr('The  has stopped unexpectedly')
