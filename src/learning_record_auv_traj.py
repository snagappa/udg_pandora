#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
import std_msgs.msg
import std_srvs.srv

import numpy

#use to load the configuration function
import cola2_ros_lib

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the valve position
from geometry_msgs.msg import PoseWithCovarianceStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from tf.transformations import euler_from_quaternion
import numpy as np

#import to use mutex
import threading
import tf

class LearningRecord:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Point()
        self.goalQuaternion = Quaternion()
        self.goalPoseOld = Point()
        self.robotPose = Odometry()
	self.initTF = False
        self.lock = threading.Lock()
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )
        self.tflistener = tf.TransformListener()


    def getConfig(self):
        param_dict = {'filename': 'learning/record/avu_traj/filename',
                      'numberSample': 'learning/record/auv_traj/number_sample',
                      'landmark_id': 'learning/record/auv_traj/landmark_id',
                      'frame_goal_id': 'learning/record/auv_traj/frame_goal_id',
                      'poseGoal_x': 'learning/record/auv_traj/poseGoal_x',
                      'poseGoal_y': 'learning/record/auv_traj/poseGoal_y',
                      'poseGoal_z': 'learning/record/auv_traj/poseGoal_z'}
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open( self.filename + "_" + str(self.numberSample) +".csv", 'w')


    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            try:
                        #Try to read the original pose detected with the visual detector
                trans, rot = self.tflistener.lookupTransform("world", self.frame_goal_id, self.tflistener.getLatestCommonTime("world","self.frame_goal_id"))
                self.goalPose.x = trans[0]
                self.goalPose.y = trans[1]
                self.goalPose.z = trans[2]
                self.goalQuaternion.x = rot[0]
                self.goalQuaternion.y = rot[1]
                self.goalQuaternion.z = rot[2]
                self.goalQuaternion.w = rot[3]
                rospy.loginfo('Goal Pose: ' + str(self.goalPose.x) +', '+ str(self.goalPose.y) +', '+ str(self.goalPose.z))

            except tf.Exception:

                for mark in landMarkMap.landmark :
                    if self.landmark_id == mark.landmark_id :
                        self.goalPose = mark.position
                        self.goalQuaternion = mark.orientation
                        rospy.loginfo('Goal Pose: ' + str(self.goalPose.x) +', '+ str(self.goalPose.y) +', '+ str(self.goalPose.z))

        finally:
            self.lock.release()


    def updateRobotPose (self, odometry):
        self.lock.acquire()
        try:
            #self.robotPose = odometry
            goalYaw = euler_from_quaternion(self.goalQuaternion)[2]
            robotYaw = odometry.twist.twist.angular[2]
            s = repr(odometry.pose.pose.Point.x - self.goalPose.x )+" "+ repr( odometry.pose.pose.Point.y - self.goalPose.y ) + " " + repr( odometry.pose.pose.Point.z - self.goalPose.z ) +" "+ repr(robotYaw-goalYaw)  +"\n"

        finally:
            self.lock.release()
        self.file.write(s)

if __name__ == '__main__':
    try:
        rospy.init_node('learning_record_auv_traj')
        learning_record = LearningRecord( rospy.get_name() )
        rospy.spin()
    except rospy.ROSInterruptException: pass
