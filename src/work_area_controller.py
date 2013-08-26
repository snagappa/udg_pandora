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
from geometry_msgs.msg import Pose

#include the float message
from std_msgs.msg import Float64

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map

#include the message to send velocities to the robot
from auv_msgs.msg import BodyVelocityReq

from std_srvs.srv import Empty, EmptyResponse

#import to use mutex
import threading
import tf


class WorkAreaController:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Pose()
        self.robotPose = Odometry()
#        self.armPose = PoseStamped()
        self.initTF = False
        self.lock = threading.Lock()
#        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry,
                         self.updateRobotPose)
        self.pub_decision = rospy.Publisher('/work_area/evaluation', Float64)
        self.tflistener = tf.TransformListener()
        self.pub_auv_vel = rospy.Publisher("/cola2_control/body_velocity_req",
                                           BodyVelocityReq)

#        self.arm_pose_init = False
        self.robot_pose_init = False
        self.goal_pose_init = False
        self.enabled = False

        self.distance_goal = 0.0
        self.gamma = 0.0
        self.alpha = 0.0
        self.beta = 0.0

        self.enable_srv = rospy.Service(
            '/learning/enable_work_area',
            Empty,
            self.enableSrv)

        self.disable_srv = rospy.Service(
            '/learning/disable_work_area',
            Empty,
            self.disableSrv)

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
                      'k_x': 'work_area/k_x',
                      'k_y': 'work_area/k_y',
                      'k_z': 'work_area/k_z',
                      'k_yaw': 'work_area/k_yaw'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.pose.pose
                    rot_matrix = tf.transformations.quaternion_matrix(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])
                    # rospy.loginfo('Update ' +
                    #               str(self.goalPose.orientation.x)
                    #               + ', ' + str(self.goalPose.orientation.y)
                    #               + ', ' + str(self.goalPose.orientation.z)
                    #               + ', ' +
                    #               str(self.goalPose.orientation.w))

                    self.goalPoseTest = mark.pose.pose

                    goalPose = np.asarray([self.poseGoal_x,
                                           self.poseGoal_y,
                                           self.poseGoal_z,
                                           1])
                    goalPose_rot = np.dot(rot_matrix, goalPose)

                    self.goalPose.position.x = (self.goalPose.position.x +
                                                goalPose_rot[0])
                    self.goalPose.position.y = (self.goalPose.position.y +
                                                goalPose_rot[1])
                    self.goalPose.position.z = (self.goalPose.position.z +
                                                goalPose_rot[2])
                    self.goal_pose_init = True
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
            self.robot_pose_init = True
        finally:
            self.lock.release()

    def enableSrv(self, req):
        self.enabled = True
        rospy.loginfo('%s Enabled', self.name)
        return EmptyResponse()

    def disableSrv(self, req):
        self.enabled = False
        rospy.loginfo('%s Disabled', self.name)
        return EmptyResponse()

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
            np.power(self.robotPose.pose.pose.position.x -
                     self.goalPose.position.x, 2) +
            np.power(self.robotPose.pose.pose.position.y -
                     self.goalPose.position.y, 2) +
            np.power(self.robotPose.pose.pose.position.z -
                     self.goalPose.position.z, 2))
        #rospy.loginfo('Distance :' + str(self.distance_goal))
        try:
            #compute the angle gamma
            trans_p, rot_p = self.tflistener.lookupTransform(
                'girona500', self.frame_goal_id,
                self.tflistener.getLatestCommonTime(
                    'girona500', self.frame_goal_id))
            # rospy.loginfo('New gamma :' +
            #               str(np.arctan2(trans_p[1], trans_p[0])))
            # rospy.loginfo('Values trans ' + str(trans_p[0])
            #               + ', ' + str(trans_p[1])
            #               + ', ' + str(trans_p[2]))
            rotation_matrix = tf.transformations.quaternion_matrix(rot_p)
            valve_pose = np.array([self.poseGoal_x,
                                   self.poseGoal_y,
                                   self.poseGoal_z,
                                   1])
            valve_pose_rot = np.dot(rotation_matrix, valve_pose)
            # rospy.loginfo('Values rotated ' + str(valve_pose_rot[0])
            #               + ', ' + str(valve_pose_rot[1])
            #               + ', ' + str(valve_pose_rot[2]))

            self.gamma = np.arctan2(trans_p[1] + valve_pose_rot[1],
                                    trans_p[0] + valve_pose_rot[0])
        except tf.Exception:
            rospy.logerr('Gamma not computed')
            self.gamma = -99.0
            # rot_matrix = tf.transformations.quaternion_matrix(
            #     [self.robotPose.pose.pose.orientation.x,
            #      self.robotPose.pose.pose.orientation.y,
            #      self.robotPose.pose.pose.orientation.z,
            #      self.robotPose.pose.pose.orientation.w])

            # trans_matrix_test = tf.transformations.translation_matrix(
            #     [-self.robotPose.pose.pose.position.x,
            #      -self.robotPose.pose.pose.position.y,
            #      -self.robotPose.pose.pose.position.z])

            # valve_pose_test = np.array(
            #     [self.goalPose.position.x,
            #      self.goalPose.position.y,
            #      self.goalPose.position.z,
            #      1])

            # # #rospy.loginfo('Translation Matrix ' + str(trans_matrix_test))
            # valve_pose_tr = np.dot(trans_matrix_test, valve_pose_test)

            # # #rospy.loginfo('Translation Result ' + str(valve_pose_tr))

            # valve_pose_rot = np.dot(rot_matrix, valve_pose_tr)

            # rospy.loginfo('Trans Manual ' + str(valve_pose_rot[0]) + ', '
            #               + str(valve_pose_rot[1]) + ', ' +
            #               str(valve_pose_rot[2]))

            # self.gamma = np.arctan2(valve_pose_rot[1],
            #                         valve_pose_rot[0])
        #rospy.loginfo('Gamma ' + str(self.gamma))
        try:
            #compute the angles alpha and beta
            trans_r, rot_r = self.tflistener.lookupTransform(
                self.frame_goal_id,
                'girona500',
                self.tflistener.getLatestCommonTime(
                    self.frame_goal_id, 'girona500'))
            # rospy.loginfo('Translation Direct ' +
            #               str(trans_r[0] + self.poseGoal_x) + ', '
            #               + str(trans_r[1] + self.poseGoal_y) + ', '
            #               + str(trans_r[2] + self.poseGoal_z))

            self.alpha = np.arctan2(trans_r[0] + self.poseGoal_x,
                                    trans_r[2] + self.poseGoal_z)
            self.beta = np.arctan2(trans_r[1] + self.poseGoal_y,
                                   trans_r[2] + self.poseGoal_z)
        except tf.Exception:
            rospy.loginfo('Aplha and Beta are not computed')
            # dist_g500_goal = [(self.robotPose.pose.pose.position.x -
            #                    self.goalPose.position.x),
            #                   (self.robotPose.pose.pose.position.y -
            #                    self.goalPose.position.y),
            #                   (self.robotPose.pose.pose.position.z -
            #                    self.goalPose.position.z),
            #                   1]
            # rotation_matrix = tf.transformations.quaternion_matrix(
            #     [self.quaternion_x,
            #      self.quaternion_y,
            #      self.quaternion_z,
            #      self.quaternion_w])
            # rotation_matrix = np.linalg.inv(rotation_matrix)
            # #trans = tf.transformations.translation_matrix(trans_g500)
            # robot_pose_rot = np.dot(rotation_matrix, dist_g500_goal)[:3]
            # #rospy.loginfo('Robot Pose from Valve ' + str(robot_pose_rot) )
            # self.alpha = np.arctan2(robot_pose_rot[0], robot_pose_rot[2])
            # self.beta = np.arctan2(robot_pose_rot[1], robot_pose_rot[2])
            self.alpha = -99.0
            self.beta = -99.0

        #rospy.loginfo('Alpha ' + str(self.alpha))
        #rospy.loginfo('Beta ' + str(self.beta))
#        rospy.loginfo('Robot Position' + str([self.robotPose.pose.pose.position.x, self.robotPose.pose.pose.position.y, self.robotPose.pose.pose.position.z ]) )

    #check if the position computed is in the limits
    def evaluatePosition(self):
        #magic stuff with the limtis
        #define manually the sigma and mu
        #possible values mu

        #p(x) =  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)
        # if is outsite the working area
        if ((self.distance_goal < self.limit_distance_goal[0] or
             self.distance_goal > self.limit_distance_goal[1])
            or np.abs(self.gamma) > self.limit_gamma
            or np.abs(self.alpha) > self.limit_alpha
            or (self.beta < self.limit_beta[0] or
                self.beta > self.limit_beta[1])):
            ## Debug code show messages
            # rospy.loginfo('Distance Goal ' +
            #               str(self.distance_goal) +
            #               ' Min and Max ' +
            #               str(self.limit_distance_goal) + ' ' +
            #               str((self.distance_goal < self.limit_distance_goal[0] or
            #                    self.distance_goal > self.limit_distance_goal[1])))
            # rospy.loginfo('Gamma ' +
            #               str(self.gamma) +
            #               ' Min and Max ' +
            #               str(self.limit_gamma) + ' ' +
            #               str(np.abs(self.gamma) > self.limit_gamma))
            # rospy.loginfo('Alpha ' +
            #               str(self.alpha) +
            #               ' Min and Max ' +
            #               str(self.limit_alpha) + ' ' +
            #               str(np.abs(self.alpha) > self.limit_alpha))
            # rospy.loginfo('Beta ' +
            #               str(self.beta) +
            #               ' Min and Max ' +
            #               str(self.limit_beta) + ' ' +
            #               str(self.beta < self.limit_beta[0] or
            #                   self.beta > self.limit_beta[1]))
            #This code move the auv to the desired position
            rospy.loginfo(
                'The robot is outsite of the working area')

            # vel_com = BodyVelocityReq()
            # vel_com.header.stamp = rospy.get_rostime()
            # vel_com.goal.priority = 10
            # #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
            # vel_com.goal.requester = 'work_area_controller'
            # #maybe with a PID it will be better
            # vel_com.twist.linear.x = self.k_x*(1.26 - self.distance_goal)
            # vel_com.twist.linear.y = self.k_y*(0.0 - self.alpha)
            # vel_com.twist.linear.z = self.k_z*(0.1 - self.beta)
            # vel_com.twist.angular.z = self.k_yaw*(0.0 - self.gamma)

            # #disabled_axis boby_velocity_req
            # vel_com.disable_axis.x = False
            # vel_com.disable_axis.y = False
            # vel_com.disable_axis.z = False
            # vel_com.disable_axis.roll = True
            # vel_com.disable_axis.pitch = True
            # vel_com.disable_axis.yaw = False

            # self.pub_auv_vel.publish(vel_com)
            if self.distance_goal > self.limit_distance_goal[1]:
                #The robot is far from the panel
                # New range [0.2..1.0]
                # Distance range [limit_dist_goal .. 5.0]
                error = -1*((((1.0 - 0.2) *
                              (self.distance_goal -
                               self.limit_distance_goal[1]))
                             / (5.0 - self.limit_distance_goal[1]))+0.2)
            elif self.distance_goal < self.limit_distance_goal[0]:
                #The robot is to close to the robot
                error = -0.25
            else:
                #The robot is not in the correct position we wait
                # the controller estavilize the orientation.
                error = 0.0
            return error
        else:
#            rospy.loginfo('The robot is insite the working area')
            return 1.0

    def run(self):
        while not rospy.is_shutdown():
            if self.enabled:
                if self.robot_pose_init and self.goal_pose_init:
                    self.computePosition()
                    safe_rate = self.evaluatePosition()
                    #publish the rate to modify the behaviour 
                    #of the learning program
                    self.pub_decision.publish(safe_rate)
                else:
                    rospy.loginfo('Waiting to initialize all the positions')
            rospy.sleep(self.period)


if __name__ == '__main__':
    try:
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "work_area_controller.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate work_area_controller.yaml")

        rospy.init_node('work_area_controller')
        work_area_controller = WorkAreaController(rospy.get_name())
        work_area_controller.run()
    except rospy.ROSInterruptException:
        pass
