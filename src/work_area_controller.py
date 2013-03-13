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

#include the message to send velocities to the robot
from auv_msgs.msg import BodyVelocityReq

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
#        self.armPose = PoseStamped()
        self.initTF = False
        self.lock = threading.Lock()
#        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )
        self.pub_decision = rospy.Publisher('/work_area/evaluation', Float64)
        self.tflistener = tf.TransformListener()
        self.pub_auv_vel = rospy.Publisher("/cola2_control/body_velocity_req", BodyVelocityReq)

#        self.arm_pose_init = False
        self.robot_pose_init = False
        self.goal_pose_init = False

        self.distance_goal = 0.0
        self.gamma = 0.0
        self.alpha = 0.0
        self.beta = 0.0

    def getConfig(self):
        param_dict = {'limit_distance_goal': 'work_area/distance_goal',
                      'limit_gamma': 'work_area/gamma',
                      'limit_alpha': 'work_area/alpha',
                      'limit_beta': 'work_area/beta',
                      'landmark_id': 'work_area/landmark_id',
                      'frame_goal_id': 'work_area/frame_goal_id',
                      'period': 'work_area/period',
                      'poseGoal_x': 'work_area/poseGoal_x',
                      'poseGoal_y': 'work_area/poseGoal_y',
                      'poseGoal_z': 'work_area/poseGoal_z',
                      'quaternion_x': 'work_area/quaternion_x',
                      'quaternion_y': 'work_area/quaternion_y',
                      'quaternion_z': 'work_area/quaternion_z',
                      'quaternion_w': 'work_area/quaternion_w',
                      'k_x' : 'work_area/k_x',
                      'k_y' : 'work_area/k_y',
                      'k_z' : 'work_area/k_z',
                      'k_yaw' : 'work_area/k_yaw'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)


    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:

            for mark in landMarkMap.landmark :
                if self.landmark_id == mark.landmark_id :
                    try:
                        trans, rot = self.tflistener.lookupTransform("world", "panel_centre", self.tflistener.getLatestCommonTime("world", "panel_centre" ))
                        rotation_matrix = tf.transformations.quaternion_matrix(rot)

                        goalPose = np.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose )

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        self.goalQuaternion.x = rot[0]
                        self.goalQuaternion.y = rot[1]
                        self.goalQuaternion.z = rot[2]
                        self.goalQuaternion.w = rot[3]

                    except tf.Exception :
                        rotation_matrix = tf.transformations.quaternion_matrix([self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w])
                        goalPose = np.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose )

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        self.goalQuaternion.x = self.quaternion_x
                        self.goalQuaternion.y = self.quaternion_y
                        self.goalQuaternion.z = self.quaternion_z
                        self.goalQuaternion.w = self.quaternion_w

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

#Not need for the moment. Only use the valve an

    # def updateArmPose(self, armPose):
    #     self.lock.acquire()
    #     try:
    #         self.armPose = armPose
    #         self.arm_pose_init = True
    #     finally:
    #         self.lock.release()

    #using the analytical function generate the
    #pitch yaw and pose of the robot
    def computePosition(self):
        #distance compute euclidean distance
        self.distance_goal = np.sqrt(
            np.power(self.robotPose.pose.pose.position.x - self.goalPose.x, 2) +
            np.power(self.robotPose.pose.pose.position.y - self.goalPose.y, 2) +
            np.power(self.robotPose.pose.pose.position.z - self.goalPose.z, 2)
            )
#        rospy.loginfo('Distance :' + str(self.distance_goal) )

        try:
            #compute the angle gamma
            goal_pose_world = np.array([self.goalPose.x,
                                        self.goalPose.y,
                                        self.goalPose.z,
                                        1])

            trans_g500 , rot_g500 = self.tflistener.lookupTransform("girona500", "world", self.tflistener.getLatestCommonTime("girona500","world" ))
            rotation_matrix = tf.transformations.quaternion_matrix(rot_g500)
            translation_matrix = tf.transformations.translation_matrix(trans_g500)

            goal_pose_rot = np.dot(rotation_matrix, goal_pose_world )
            goal_pose_rot = np.dot(translation_matrix, goal_pose_rot )
            #has not sense, doesppn't work correctly
            self.gamma = np.arctan2( goal_pose_rot[1], goal_pose_rot[0] )
        except tf.Exception:
            rospy.logerr('Gamma not computed')
            self.gamma = 0.0

#        rospy.loginfo('Gamma ' + str(self.gamma) )

        try:
            #compute the angles alpha and beta
            robot_pose_world = np.array([self.robotPose.pose.pose.position.x,
                                         self.robotPose.pose.pose.position.y,
                                         self.robotPose.pose.pose.position.z,
                                         1])

            trans_goal, rot_goal = self.tflistener.lookupTransform('panel_centre', 'world',  self.tflistener.getLatestCommonTime('panel_centre','world'))

            rotation_matrix = tf.transformations.quaternion_matrix(rot_goal)
            translation_matrix = tf.transformations.translation_matrix(trans_goal)
            robot_pose_rot = np.dot(rotation_matrix, robot_pose_world)
            robot_pose = np.dot(translation_matrix, robot_pose_rot)

            self.alpha = np.arctan2( robot_pose[0], robot_pose[2] )
            self.beta = np.arctan2( robot_pose[1], robot_pose[2] )
        except tf.Exception:
            rospy.loginfo('Aplha and Beta Estimated')
            dist_g500_goal = [self.robotPose.pose.pose.position.x - self.goalPose.x,
                              self.robotPose.pose.pose.position.y - self.goalPose.y ,
                              self.robotPose.pose.pose.position.z - self.goalPose.z ,
                              1]
            rotation_matrix = tf.transformations.quaternion_matrix(
                [ self.quaternion_x,
                  self.quaternion_y,
                  self.quaternion_z,
                  self.quaternion_w ] )
            rotation_matrix = np.linalg.inv(rotation_matrix)
            #trans = tf.transformations.translation_matrix(trans_g500)
            robot_pose_rot = np.dot(rotation_matrix, dist_g500_goal)[:3]
            #rospy.loginfo('Robot Pose from Valve ' + str(robot_pose_rot) )
            self.alpha = np.arctan2( robot_pose_rot[0], robot_pose_rot[2] )
            self.beta = np.arctan2( robot_pose_rot[1], robot_pose_rot[2] )

#        rospy.loginfo('Alpha ' + str(self.alpha) )
#        rospy.loginfo('Beta ' + str(self.beta) )

    #check if the position computed is in the limits
    def evaluatePosition(self):
        #magic stuff with the limtis
        #define manually the sigma and mu
        #possible values mu

        #p(x) =  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)

        if ( (self.distance_goal < self.limit_distance_goal[0] or
              self.distance_goal > self.limit_distance_goal[1] )
             or np.abs(self.gamma) > self.limit_gamma
             or np.abs(self.alpha) > self.limit_alpha
             or np.abs(self.beta) > self.limit_beta ) :

            rospy.loginfo('The robot is outsite of the working area')

            vel_com = BodyVelocityReq()
            vel_com.header.stamp = rospy.get_rostime()
            vel_com.goal.priority = 10 #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
            vel_com.goal.requester = 'work_area_controller'
            #maybe with a PID it will be better
            vel_com.twist.linear.x = self.k_x*(1.28 - self.distance_goal)
            vel_com.twist.linear.y = self.k_y*(0.0 - self.alpha)
            vel_com.twist.linear.z = self.k_z*(0.0 - self.beta)
            vel_com.twist.angular.z = self.k_yaw*(0.0 - self.gamma)

            #disabled_axis boby_velocity_req
            vel_com.disable_axis.x = False
            vel_com.disable_axis.y = False
            vel_com.disable_axis.z = False
            vel_com.disable_axis.roll = True
            vel_com.disable_axis.pitch = True
            vel_com.disable_axis.yaw = False

            self.pub_auv_vel.publish(vel_com)

            return -1.0
        else :
#            rospy.loginfo('The robot is insite the working area')
            return 1.0


    def run(self):
        while not rospy.is_shutdown():
#            if self.robot_pose_init and self.arm_pose_init and self.goal_pose_init :
            if self.robot_pose_init and self.goal_pose_init :
                self.computePosition()
                safe_rate = self.evaluatePosition()
                #publish the rate to modify the behaviour of the learning program
                self.pub_decision.publish(safe_rate)
            else:
                rospy.loginfo('Waitting to initialize all the positions')

            rospy.sleep(self.period)


if __name__ == '__main__':
    try:
        rospy.init_node('work_area_controller')
        work_area_controller = WorkAreaController( rospy.get_name() )
        work_area_controller.run()
    except rospy.ROSInterruptException: pass
