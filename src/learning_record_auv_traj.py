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
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the valve position
#from geometry_msgs.msg import PoseWithCovarianceStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from tf.transformations import euler_from_quaternion
#import numpy as np

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
        rospy.Subscriber(
            "/pose_ekf_slam/odometry", Odometry, self.updateRobotPose)
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
                    self.goalPose = mark.position
                    try:
                    #Try to read the original pose detected
                    #with the visual detector
                        trans, rot = self.tflistener.lookupTransform(
                            "world",
                            self.frame_goal_id,
                            self.tflistener.getLatestCommonTime(
                                "world", self.frame_goal_id))
                        self.goalQuaternion.x = rot[0]
                        self.goalQuaternion.y = rot[1]
                        self.goalQuaternion.z = rot[2]
                        self.goalQuaternion.w = rot[3]
                        rospy.loginfo('Orientation of the panel' + str(rot))
                    except tf.Exception:
                        #rospy.loginfo('Orientation loaded from the configuration file')
                        self.goalQuaternion.x = self.quaternion_x
                        self.goalQuaternion.y = self.quaternion_y
                        self.goalQuaternion.z = self.quaternion_z
                        self.goalQuaternion.w = self.quaternion_w
                        #rospy.loginfo('Goal Pose: ' + str(self.goalPose.x) +', '+ str(self.goalPose.y) +', '+ str(self.goalPose.z))
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            #self.robotPose = odometry
            goalYaw = euler_from_quaternion(
                [self.goalQuaternion.x,
                 self.goalQuaternion.y,
                 self.goalQuaternion.z,
                 self.goalQuaternion.w])[2]
            robotYaw = euler_from_quaternion(
                [odometry.pose.pose.orientation.x,
                 odometry.pose.pose.orientation.y,
                 odometry.pose.pose.orientation.z,
                 odometry.pose.pose.orientation.w])[2]
            s = (repr(odometry.pose.pose.position.x - self.goalPose.x) +
                 " " + repr(odometry.pose.pose.position.y - self.goalPose.y) +
                 " " + repr(odometry.pose.pose.position.z - self.goalPose.z) +
                 " " + repr(cola2_lib.normalizeAngle(robotYaw-goalYaw)) + "\n")
        finally:
            self.lock.release()
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
