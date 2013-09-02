#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#use to load the configuration function
import cola2_ros_lib

#use to normalize the angle
import cola2_lib

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import Pose
#include message of the ekf giving the valve position
#from geometry_msgs.msg import PoseWithCovarianceStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
#from geometry_msgs.msg import Point
#from geometry_msgs.msg import Quaternion

#from tf.transformations import euler_from_quaternion
import numpy as np

#import to use mutex
import threading
import tf


class LearningRecord:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Pose()
        self.robotPose = Odometry()
        self.initPose = False
        self.lock = threading.Lock()
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber(
            "/pose_ekf_slam/odometry", Odometry, self.updateRobotPose)
        # Frequenzy 30 Hz
        self.tflistener = tf.TransformListener()

    def getConfig(self):
        param_dict = {'filename': 'learning/record/auv_traj/filename',
                      'numberSample': 'learning/record/auv_traj/number_sample',
                      'landmark_id': 'learning/record/auv_traj/landmark_id',
                      'frame_goal_id': 'learning/record/auv_traj/frame_goal_id',
                      'quaternion_x': 'learning/record/auv_traj/quaternion_x',
                      'quaternion_y': 'learning/record/auv_traj/quaternion_y',
                      'quaternion_z': 'learning/record/auv_traj/quaternion_z',
                      'quaternion_w': 'learning/record/auv_traj/quaternion_w'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open(
            self.filename + "_" + str(self.numberSample) + ".csv", 'w')

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.pose.pose
                    self.initPose = True
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            #self.robotPose = odometry
            # rot_mat = tf.transformations.quaternion_matrix(
            #     [self.goalPose.orientation.x,
            #      self.goalPose.orientation.y,
            #      self.goalPose.orientation.z,
            #      self.goalPose.orientation.w])

            # trans_mat = tf.transformations.translation_matrix(
            #     [self.goalPose.position.x,
            #      self.goalPose.position.y,
            #      self.goalPose.position.z,
            #      1])

            # robotPose = np.array([odometry.pose.pose.position.x,
            #                       odometry.pose.pose.position.y,
            #                       odometry.pose.pose.position.z])

            trans_mat = tf.transformations.quaternion_matrix(
                [odometry.pose.pose.orientation.x,
                 odometry.pose.pose.orientation.y,
                 odometry.pose.pose.orientation.z,
                 odometry.pose.pose.orientation.w])

            # trans_mat = tf.transformations.translation_matrix(
            #     [odometry.pose.pose.position.x,
            #      odometry.pose.pose.position.y,
            #      odometry.pose.pose.position.z])

            trans_mat[0, 3] = odometry.pose.pose.position.x
            trans_mat[1, 3] = odometry.pose.pose.position.y
            trans_mat[2, 3] = odometry.pose.pose.position.z

            #invert Translation Matrix
            inv_mat = np.zeros([4, 4])
            inv_mat[3, 3] = 1.0
            inv_mat[0:3, 0:3] = np.transpose(trans_mat[0:3, 0:3])
            inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]), trans_mat[0:3, 3])

            panelPose = np.array([self.goalPose.position.x,
                                  self.goalPose.position.y,
                                  self.goalPose.position.z,
                                  1])

            # rospy.loginfo('Robot Pose ' + str([odometry.pose.pose.position.x,
            #                                    odometry.pose.pose.position.y,
            #                                    odometry.pose.pose.position.z]))

            # rospy.loginfo('Panel Pose ' + str([self.goalPose.position.x,
            #                                    self.goalPose.position.y,
            #                                    self.goalPose.position.z]))

            # rospy.loginfo('Inverse Trans ' + str(np.dot(inv_mat, panelPose)))

            trans_pose = np.dot(inv_mat, panelPose)

            goalYaw = tf.transformations.euler_from_quaternion(
                [self.goalPose.orientation.x,
                 self.goalPose.orientation.y,
                 self.goalPose.orientation.z,
                 self.goalPose.orientation.w])[2]
            robotYaw = tf.transformations.euler_from_quaternion(
                [odometry.pose.pose.orientation.x,
                 odometry.pose.pose.orientation.y,
                 odometry.pose.pose.orientation.z,
                 odometry.pose.pose.orientation.w])[2]
            s = (repr(trans_pose[0]) +
                 " " + repr(trans_pose[1]) +
                 " " + repr(trans_pose[2]) +
                 " " + repr(cola2_lib.normalizeAngle(robotYaw-goalYaw)) + "\n")
            rospy.loginfo('Distance ' + str(trans_pose[0]) 
                          + ' ' + str(trans_pose[1])
                          + ' ' + str(trans_pose[2]))
        finally:
            self.lock.release()
        if self.initPose:
            self.file.write(s)

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_record_auv.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr(
                "Could not locate learning_record_auv.yaml")
        rospy.init_node('learning_record_auv_traj')
        learning_record = LearningRecord(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass