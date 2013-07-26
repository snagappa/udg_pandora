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
#from nav_msgs.msg import Odometry

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
        self.lock = threading.Lock()
        self.initGoalPose = False
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        #rospy.Subscriber("/pose_ekf_slam/landmark_update/valve_1",
        #                 PoseWithCovarianceStamped, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        #rospy.Subscriber("/visual_detector2/valve")
        self.tflistener = tf.TransformListener()

    def getConfig(self):
        param_dict = {'filename': 'learning/record/filename',
                      'numberSample': 'learning/record/number_sample',
                      'landmark_id': 'learning/record/landmark_id',
                      'frame_goal_id': 'learning/record/frame_goal_id',
                      'poseGoal_x': 'learning/record/poseGoal_x',
                      'poseGoal_y': 'learning/record/poseGoal_y',
                      'poseGoal_z': 'learning/record/poseGoal_z',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open(self.filename + "_" +
                         str(self.numberSample) + ".csv", 'w')

    def updateArmPose(self, armPose):
        #quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y,
        #              armPose.pose.orientation.z, armPose.pose.orientation.w]
        #euler = euler_from_quaternion(quaternion, 'sxyz')
        self.lock.acquire()
        try:
            if self.initGoalPose:
                try:
                    arm_pose, rot = self.tflistener.lookupTransform(
                        "world", "end_effector",
                        self.tflistener.getLatestCommonTime(
                            "world", "end_effector"))
                    #rospy.loginfo( 'Arm global Pose ' + str(arm_pose)  )
                    arm_ori = euler_from_quaternion(rot)
                    #In the angles between the end EE and the valve are changed
                    # Roll is the difference between Pitch in the world
                    # Pitch is the difference between Roll in the world
                    # Yaw is the differences between the Yaw in the world
                    s = (repr(arm_pose[0] - self.goalPose.position.x) + " " +
                         repr(arm_pose[1] - self.goalPose.position.y) + " " +
                         repr(arm_pose[2] - self.goalPose.position.z) + " " +
                         repr(cola2_lib.normalizeAngle(arm_ori[1] -
                                                       self.goalOrientation[1]))
                         + " " +
                         repr(cola2_lib.normalizeAngle(
                             cola2_lib.normalizeAngle(arm_ori[0] -
                                                      self.goalOrientation[0])
                             - (math.pi/2.0)))
                         + " " +
                         repr(cola2_lib.normalizeAngle(arm_ori[2] -
                                                       self.goalOrientation[2]))
                         + "\n")
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
        finally:
            self.lock.release()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_record.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_record.yaml")

        rospy.init_node('learning_record')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record = LearningRecord(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
