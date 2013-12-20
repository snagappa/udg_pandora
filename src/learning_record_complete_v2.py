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
from geometry_msgs.msg import PoseWithCovarianceStamped
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
        self.initGoal = False
        self.initGoalPose = False
        self.initGoalOri = False
        self.initRobotPose = False
        self.valveOri = 0.0
        self.valveOriInit = False

        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)

        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalOri)

        rospy.Subscriber(
            "/pose_ekf_slam/odometry", Odometry, self.updateRobotPose)

        rospy.Subscriber("/valve_tracker/valve" + str(self.goal_valve),
                         PoseWithCovarianceStamped,
                         self.updateGoalPose)
        self.tflistener = tf.TransformListener()

    def getConfig(self):
        param_dict = {'filename': 'learning/record/complete/filename',
                      'numberSample': 'learning/record/complete/number_sample',
                      'landmark_id': 'learning/record/complete/landmark_id',
                      'frame_goal_id': 'learning/record/complete/frame_goal_id',
                      'poseGoal_x': 'learning/record/complete/poseGoal_x',
                      'poseGoal_y': 'learning/record/complete/poseGoal_y',
                      'poseGoal_z': 'learning/record/complete/poseGoal_z',
                      'goal_valve': 'learning/record/complete/goal_valve',
                      'base_pose': '/arm_controller/base_pose',
                      'base_ori': '/arm_controller/base_ori'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open(self.filename + "_" +
                         str(self.numberSample) + ".csv", 'w')

    def updateGoalPose(self, pose_msg):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param pose_msg: Contains the position and the orientation of the vavle
        @type pose_msg: PoseWithCovarianceStamped
        """
        self.lock.acquire()
        try:
            self.goalPose.position = pose_msg.pose.pose.position
            self.valveOri = euler_from_quaternion(
                            [self.goalPose.orientation.x,
                             self.goalPose.orientation.y,
                             self.goalPose.orientation.z,
                             self.goalPose.orientation.w])[2]
            if not self.initGoalPose:
                self.initGoalPose = True
                if (self.initGoalOri and
                    not self.initGoal):
                    self.initGoal = True
            if not self.valveOriInit:
                self.valveOriInit = True
        finally:
            self.lock.release()

    def updateGoalOri(self, landMarkMap):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param landMarkMap: Contains the position and the orientation of the vavle and panel
        @type landMarkMap: Map with
        """
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose.orientation = mark.pose.pose.orientation
                    if not self.initGoalOri:
                        self.initGoalOri = True
                        if (self.initGoalPose and
                            not self.initGoal):
                            self.initGoal = True
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        """
        This method update the position of the robot. Using directly the pose
        published by the pose_ekf_slam.
        @param odometry: Contains the odometry computed in the pose_ekf
        @type odometry: Odometry message
        """
        self.lock.acquire()
        try:
            self.robotPose = odometry.pose.pose
            self.initRobotPose = True
        finally:
            self.lock.release()

    def updateArmPose(self, armPose):
        """
        This method update the pose of the end-effector using as a frame center
        the base of the manipulator. Also Compute the position of the AUV and
        end-effector using as a frame center the position of the panel. Finally
        it writes the positions in a csv file.
        @param armPose: Contains the position and orientation of the End-effector respect the base of the arm
        @type armPose: PoseStamped
        """
        #quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y,
        #              armPose.pose.orientation.z, armPose.pose.orientation.w]
        #euler = euler_from_quaternion(quaternion, 'sxyz')
        self.lock.acquire()
        try:
            if self.initGoalPose:
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
                # End-Effector Pose from the Base_arm without TF
                #################################################

                #transformation from the world to the robot
                trans_matrix_v2 = tf.transformations.quaternion_matrix(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])

                trans_matrix_v2[0, 3] = self.robotPose.position.x
                trans_matrix_v2[1, 3] = self.robotPose.position.y
                trans_matrix_v2[2, 3] = self.robotPose.position.z

                arm_pose = np.array([armPose.pose.position.x,
                                     armPose.pose.position.y,
                                     armPose.pose.position.z,
                                     1])

                robot_base = tf.transformations.euler_matrix(
                    self.base_ori[0],
                    self.base_ori[1],
                    self.base_ori[2])

                robot_base[0, 3] = self.base_pose[0]
                robot_base[1, 3] = self.base_pose[1]
                robot_base[2, 3] = self.base_pose[2]

                arm_base = np.dot(robot_base, arm_pose)

                arm_world_pose = np.dot(trans_matrix_v2, arm_base)

                arm_frame_pose = np.dot(inv_mat, arm_world_pose)

                arm_ori = euler_from_quaternion([armPose.pose.orientation.x,
                                                 armPose.pose.orientation.y,
                                                 armPose.pose.orientation.z,
                                                 armPose.pose.orientation.w])

                #Wrong orientation is not correct.
                if self.valveOriInit:
                    roll = self.valveOri - arm_ori[2]
                else:
                    roll = arm_ori[2]

                s = (repr(robotTrans[0]) + " " +
                     repr(robotTrans[1]) + " " +
                     repr(robotTrans[2]) + " " +
                     repr(cola2_lib.normalizeAngle(goalYaw - robotYaw))
                     + " " +
                     repr(arm_frame_pose[0]) + " " +
                     repr(arm_frame_pose[1]) + " " +
                     repr(arm_frame_pose[2]) + " " +
                     repr(arm_ori[0]) + " " +
                     repr(arm_ori[1]) + " " +
                     repr(roll) + " " +
                     repr(rospy.get_time()) + "\n")
                self.file.write(s)
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