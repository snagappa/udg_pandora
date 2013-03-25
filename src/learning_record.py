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
        self.goalPoseOld = Point()
        self.goalQuaternion = Quaternion()

        self.robotPose = Odometry()
	self.initTF = False
        self.lock = threading.Lock()
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        #rospy.Subscriber("/pose_ekf_slam/landmark_update/valve_1", PoseWithCovarianceStamped, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        #rospy.Subscriber("/visual_detector2/valve")
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )
        self.tflistener = tf.TransformListener()
       # self.record = 0

    def getConfig(self):
        param_dict = {'filename': 'learning/record/filename',
                      'numberSample': 'learning/record/number_sample',
                      'landmark_id': 'learning/record/landmark_id',
                      'frame_goal_id': 'learning/record/frame_goal_id',
                      'poseGoal_x': 'learning/record/poseGoal_x',
                      'poseGoal_y': 'learning/record/poseGoal_y',
                      'poseGoal_z': 'learning/record/poseGoal_z',
                      'quaternion_x': 'learning/record/quaternion_x',
                      'quaternion_y': 'learning/record/quaternion_y',
                      'quaternion_z': 'learning/record/quaternion_z',
                      'quaternion_w': 'learning/record/quaternion_w'
}
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open( self.filename + "_" + str(self.numberSample) +".csv", 'w')

    def updateArmPose(self, armPose):
#        euler = euler_from_quaternion( armPose.pose.orientation, 'sxyz' )  #,  axes='sxyz' );
        quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y, armPose.pose.orientation.z, armPose.pose.orientation.w ]
        euler = euler_from_quaternion( quaternion, 'sxyz' )
        #s = repr(armPose.pose.position.x)+" "+ repr(armPose.pose.position.y) + " " + repr(armPose.pose.position.x) +" "+ repr(euler[0])  +" "+ repr(euler[1])  +" "+ repr(euler[2]) +"\n"

        self.lock.acquire()
        try:
            arm_pose, rot = self.tflistener.lookupTransform("world", "end_effector", self.tflistener.getLatestCommonTime("world","end_effector"))

            #rospy.loginfo( 'Arm global Pose ' + str(arm_pose)  )

            # trans, rot = self.tflistener.lookupTransform("world", "girona500", self.tflistener.getLatestCommonTime("world","girona500"))
            # rotation_matrix = tf.transformations.quaternion_matrix(rot)
            # arm_pose = np.asarray([armPose.pose.position.x, armPose.pose.position.y, armPose.pose.position.z, 1])
            # arm_pose_tf = np.dot(rotation_matrix, arm_pose)[:3]

            s = repr(arm_pose[0] - self.goalPose.x )+" "+ repr( arm_pose[1] - self.goalPose.y ) + " " + repr( arm_pose[2] - self.goalPose.z ) +" "+ repr(rot[0])  +" "+ repr(rot[1])  +" "+ repr(rot[2]) + " " + repr(rot[3])+"\n"

            # rospy.loginfo( 'Arm robot Pose: ' + str(arm_pose_tf[0]) )
            # rospy.loginfo( 'Arm robot pose : ' + str(armPose.pose.position.x) )
            # rospy.loginfo( 'Robot global pose : ' + str(self.robotPose.pose.pose.position.x) )

            #rospy.loginfo( 'Arm global Pose: ' + str(arm_pose_tf[0] + self.robotPose.pose.pose.position.x ) +', ' + str(arm_pose_tf[1] + self.robotPose.pose.pose.position.y ) +', ' + str(arm_pose_tf[2] + self.robotPose.pose.pose.position.z ))

            #rospy.loginfo('Valve centre global pose: ' + str(self.goalPose.x ) +', ' + str(self.goalPose.y ) +', ' +  str(self.goalPose.z ))

            #rospy.loginfo('Distance Arm Valve' + str(arm_pose_tf[0] - self.goalPose.x) +', ' + str(arm_pose_tf[1] - self.goalPose.y) +', ' + str(arm_pose_tf[2] - self.goalPose.z) )

        finally:
            self.lock.release()

        self.file.write(s)
        # if ( self.record == 10 ) :
        #     self.file.write(s)
        #     self.record = 0
        # else :
        #     self.record += 1

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:

            for mark in landMarkMap.landmark :
                if self.landmark_id == mark.landmark_id :
                    self.goalPose = mark.position
                    try:
                        trans, rot = self.tflistener.lookupTransform("world", "panel_centre", self.tflistener.getLatestCommonTime("world", "panel_centre" ))
                        rotation_matrix = tf.transformations.quaternion_matrix(rot)

                        goalPose = numpy.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = numpy.dot(rotation_matrix, goalPose )

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        self.goalQuaternion.x = rot[0]
                        self.goalQuaternion.y = rot[1]
                        self.goalQuaternion.z = rot[2]
                        self.goalQuaternion.w = rot[3]

                    except tf.Exception :
                        rotation_matrix = tf.transformations.quaternion_matrix([self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w])
                        goalPose = numpy.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = numpy.dot(rotation_matrix, goalPose )

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        self.goalQuaternion.x = self.quaternion_x
                        self.goalQuaternion.y = self.quaternion_y
                        self.goalQuaternion.z = self.quaternion_z
                        self.goalQuaternion.w = self.quaternion_w
        finally:
            self.lock.release()
    def updateRobotPose (self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
        finally:
            self.lock.release()


if __name__ == '__main__':
    try:
        rospy.init_node('learning_record')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record = LearningRecord( rospy.get_name() )
        rospy.spin()
    except rospy.ROSInterruptException: pass
