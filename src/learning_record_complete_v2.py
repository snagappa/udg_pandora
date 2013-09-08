#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

import numpy as np
#use to load the configuration function
import cola2_ros_lib
#use to normalize the angle
import cola2_lib
#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the valve position
#from geometry_msgs.msg import PoseWithCovarianceStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Pose
#from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion
#import to use mutex
import threading
import tf
import math


class LearningRecord:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        rospy.loginfo('Configuration Loaded')
        self.goalPose = Pose()
        self.robotPose = Pose()
        self.lock = threading.Lock()
        self.initTime = 0.0
        self.initGoalPose = False
        self.initRobotPose = False
        self.valveOri = 0.0
        self.valveOriInit = False

        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)

        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)

        rospy.Subscriber(
            "/pose_ekf_slam/odometry", Odometry, self.updateRobotPose)

        self.tflistener = tf.TransformListener()

    def getConfig(self):
        param_dict = {'filename': 'learning/record/complete/filename',
                      'numberSample': 'learning/record/complete/number_sample',
                      'landmark_id': 'learning/record/complete/landmark_id',
                      'frame_goal_id': 'learning/record/complete/frame_goal_id',
                      'poseGoal_x': 'learning/record/complete/poseGoal_x',
                      'poseGoal_y': 'learning/record/complete/poseGoal_y',
                      'poseGoal_z': 'learning/record/complete/poseGoal_z',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open(self.filename + "_" +
                         str(self.numberSample) + ".csv", 'w')

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.pose.pose
                    try:
                        trans, rot = self.tflistener.lookupTransform(
                            "world", "panel_centre",
                            self.tflistener.getLatestCommonTime(
                                "world", "panel_centre"))
                        rotation_matrix = tf.transformations.quaternion_matrix(
                            rot)
                        goalPose = np.asarray([self.poseGoal_x,
                                               self.poseGoal_y,
                                               self.poseGoal_z,
                                               1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose)
                        self.goalPose.position.x = (self.goalPose.position.x +
                                                    goalPose_rot[0])
                        self.goalPose.position.y = (self.goalPose.position.y +
                                                    goalPose_rot[1])
                        self.goalPose.position.z = (self.goalPose.position.z +
                                                    goalPose_rot[2])
                        self.goalOrientation = euler_from_quaternion(rot)
                        self.initGoalPose = True
                    except tf.Exception:
                        rotation_matrix = tf.transformations.quaternion_matrix(
                            [self.goalPose.orientation.x,
                             self.goalPose.orientation.y,
                             self.goalPose.orientation.z,
                             self.goalPose.orientation.w])
                        #poseGoal is the position of the vavle
                        goalPose = np.asarray([self.poseGoal_x,
                                               self.poseGoal_y,
                                               self.poseGoal_z,
                                               1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose)
                        self.goalPose.position.x = (self.goalPose.position.x +
                                                    goalPose_rot[0])
                        self.goalPose.position.y = (self.goalPose.position.y +
                                                    goalPose_rot[1])
                        self.goalPose.position.z = (self.goalPose.position.z +
                                                    goalPose_rot[2])
                        self.goalOrientation = euler_from_quaternion(
                            [self.goalPose.orientation.x,
                             self.goalPose.orientation.y,
                             self.goalPose.orientation.z,
                             self.goalPose.orientation.w])
                        self.initGoalPose = True

                    try:
                        trans, rot = self.tflistener.lookupTransform(
                            "world", "valve2",
                            self.tflistener.getLatestCommonTime(
                                "world", "valve2"))
                        self.valveOri = tf.transformations.euler_from_quaternion(rot)[2]
                        self.valveOriInit = True
                    except tf.Exception:
                        pass
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry.pose.pose
            self.initRobotPose = True
        finally:
            self.lock.release()

    def updateArmPose(self, armPose):
        #quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y,
        #              armPose.pose.orientation.z, armPose.pose.orientation.w]
        #euler = euler_from_quaternion(quaternion, 'sxyz')
        self.lock.acquire()
        try:
            if self.initGoalPose:
                try:
                    #################################################
                    # Compute the pose of the AUV respect the Valve 2
                    #################################################
                    robotPose = np.array(
                        [self.robotPose.position.x,
                         self.robotPose.position.y,
                         self.robotPose.position.z,
                         1])

                    trans_matrix = tf.transformations.quaternion_matrix(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])

                    trans_matrix[0, 3] = self.goalPose.position.x
                    trans_matrix[1, 3] = self.goalPose.position.y
                    trans_matrix[2, 3] = self.goalPose.position.z

                    #invert Matrix
                    inv_mat = np.zeros([4, 4])
                    inv_mat[3, 3] = 1.0
                    inv_mat[0:3, 0:3] = np.transpose(trans_matrix[0:3, 0:3])
                    inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                             trans_matrix[0:3, 3])

                    robotTrans = np.dot(inv_mat, robotPose)

                    robotYaw = euler_from_quaternion(
                        [self.robotPose.orientation.x,
                         self.robotPose.orientation.y,
                         self.robotPose.orientation.z,
                         self.robotPose.orientation.w])[2]

                    goalYaw = tf.transformations.euler_from_quaternion(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])[1]

                    #################################################
                    # End-Effector Pose from the Base_arm
                    #################################################

                    pose_ef, ori_ef = self.tflistener.lookupTransform(
                        "world", "end_effector",
                        self.tflistener.getLatestCommonTime(
                            "world", "end_effector"))

                    # endeffectorPose = np.array([armPose.pose.position.x,
                    #                             armPose.pose.position.y,
                    #                             armPose.pose.position.z,
                    #                             1])

                    # armOri = euler_from_quaternion(
                    #     [armPose.pose.orientation.x,
                    #      armPose.pose.orientation.y,
                    #      armPose.pose.orientation.z,
                    #      armPose.pose.orientation.w])

                    endeffectorPose = np.array([pose_ef[0],
                                                pose_ef[1],
                                                pose_ef[2],
                                                1])

                    armOri = euler_from_quaternion(ori_ef)
                    endEfWorld = np.dot(inv_mat, endeffectorPose)

                    #Wrong orientation is not correct.
                    if self.valveOriInit:
                        roll = self.valveOri - armOri[2]
                    else:
                        roll = armOri[2]

                    s = (repr(robotTrans[0]) + " " +
                         repr(robotTrans[1]) + " " +
                         repr(robotTrans[2]) + " " +
                         repr(cola2_lib.normalizeAngle(goalYaw - robotYaw))
                         + " " +
                         repr(endEfWorld[0]) + " " +
                         repr(endEfWorld[1]) + " " +
                         repr(endEfWorld[2]) + " " +
                         repr(roll) + " " +
                         repr(armOri[1]) + " " +
                         repr(armOri[2]) + "\n")
                    self.file.write(s)
                except tf.Exception:
                    rospy.loginfo(
                        'Error in the TF using the last arm pose published')
                    #TODO Subscribe the node tho the arm position.
                    # Think how we transform for the arm position to the world position 
                    # without using the TF
            else:
                rospy.loginfo('Goal pose Not initialized')
        finally:
            self.lock.release()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_record_complete.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_record.yaml")

        rospy.init_node('learning_record_complete')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record = LearningRecord(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
