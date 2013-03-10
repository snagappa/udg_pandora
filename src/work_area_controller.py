#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

import numpy as np

#use to load the configuration function
import cola2_ros_lib

#include message to receive the position of the arm
from geometry_msgs.msg import PoseStamped

#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message for the pose of the landmark
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

#include the float message
from std_msgs.msg import Float64

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map

#import to use mutex
import threading
import tf


class WorkAreaController:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Point()
        self.goalQuaternion = Quaternion()
        self.robotPose = Odometry()
        self.armPose = PoseStamped()
        self.initTF = False
        self.lock = threading.Lock()
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )
        self.pub_decision = rospy.Publisher('/work_area/evaluation', Float64)
        self.tflistener = tf.TransformListener()

        self.arm_pose_init = False
        self.robot_pose_init = False
        self.goal_pose_init = False

    def getConfig(self):
        param_dict = {'yaw': 'work_area/yaw',
                      'pitch': 'work_area/pitch',
                      'pose_x': 'work_area/pose_x',
                      'pose_y': 'work_area/pose_y',
                      'pose_z': 'work_area/pose_z',
                      'landmark_id': 'work_area/landmark_id'
                      'frame_goal_id': 'work_area/frame_goal_id',
                      'period': 'work_area/period',
                      'goalPose_x': 'work_area/goalPose_x',
                      'goalPose_y': 'work_area/goalPose_y',
                      'goalPose_z': 'work_area/goalPose_z'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)


    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            try:
#Try to read the original pose detected with the visual detector
                trans, rot = self.tflistener.lookupTransform("world", self.frame_goal_id, self.tflistener.getLatestCommonTime("world",self.frame_goal_id))
                self.goalPose.x = trans[0]
                self.goalPose.y = trans[1]
                self.goalPose.z = trans[2]
                self.goalQuaternion.x = rot[0]
                self.goalQuaternion.y = rot[1]
                self.goalQuaternion.z = rot[2]
                self.goalQuaternion.w = rot[3]

                self.goal_pose_init = True

            except tf.Exception:
                for mark in landMarkMap.landmark :
                    if self.landmark_id == mark.landmark_id :
                        trans, rot = self.tflistener.lookupTransform("world", "panel_centre", self.tflistener.getLatestCommonTime("world", "panel_centre" ))
                        rotation_matrix = tf.transformations.quaternion_matrix(rot)
                        goalPose = numpy.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = numpy.dot(rotation_matrix, goalPose)[:3]

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        self.goalQuaternion.x = rot[0]
                        self.goalQuaternion.y = rot[1]
                        self.goalQuaternion.z = rot[2]
                        self.goalQuaternion.w = rot[3]

                        self.goal_pose_init = True

        finally:
            self.lock.release()

    def updateRobotPose (self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
            self.robot_pose_init = True
        finally:
            self.lock.release()
        self.file.write(s)

    def updateArmPose(self, armPose):
        self.lock.acquire()
        try:
            self.armPose = armPose
            self.arm_pose_init = True
        finally:
            self.lock.release()

    #using the analytical function generate the
    #pitch yaw and pose of the robot
    def computePosition(self):
        pass

    #check if the position computed is in the limits
    def evaluatePosition(self):
        #magic stuff with the limtis
        #define manually the sigma and mu
        #possible values mu

        #p(x) =  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)


        return 1.0


    def run(self):
        while not rospy.is_shutdown():
            if self.robot_pose_init and self.arm_pose_init and self.goal_pose_init :
                self.computePosition()
                safe_rate = self.evaluatePosition()
                #publish the rate to modify the behaviour of the learning program
                self.pub_decision(safe_rate)
            else:
                rospy.info('Waitting to initialize all the positions')

            rospy.sleep(self.period)


if __name__ == '__main__':
    try:
        rospy.init_node('learning_record_auv_traj')
        learning_record = LearningRecord( rospy.get_name() )
        learning_record.run()
    except rospy.ROSInterruptException: pass
