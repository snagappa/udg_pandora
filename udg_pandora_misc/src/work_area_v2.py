#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora_misc')
import rospy

import numpy as np

#use to load the configuration function
import cola2_ros_lib

#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message for the pose of the landmark
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map

#include the message to send velocities to the robot
from learning_pandora.msg import rfdm_msg

#import to use mutex
import threading
import tf

class WorkAreaController:
    """
    This class evaluates the safety of the manipulation using 4 paramters.
    Computing the diference between the base of the arm and the desired valve
    """


    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Pose()
        self.robotPose = Pose()
#        self.armPose = PoseStamped()
        self.lock = threading.Lock()

        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.updateGoalOri,
                         queue_size = 1)

        rospy.Subscriber(
            "/pose_ekf_slam/odometry",
            Odometry,
            self.updateRobotPose,
            queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve" + str(self.goal_valve),
                         PoseWithCovarianceStamped,
                         self.updateGoalPose,
                         queue_size = 1)

        self.pub_decision = rospy.Publisher('/rfdm_pkg/reactive', rfdm_msg)

#        self.arm_pose_init = False
        self.initRobotPose = False
        self.initGoalPose = False
        self.initGoalOri = False
        self.initGoal = False

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
                      'period': 'work_area/period',
                      'goal_valve': 'work_area/goal_valve',
                      'base_pose': '/arm_controller/base_pose',
                      'base_ori': '/arm_controller/base_ori',
                      'limit_enter_zone' : 'work_area/limit_enter_zone',
                      'limit_wait_outsite' : 'work_area/limit_wait_outsite',
                      'limit_outsite' : 'work_area/limit_outsite'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)


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
            if not self.initGoalPose:
                self.initGoalPose = True
                if (self.initGoalOri and
                    not self.initGoal):
                    self.initGoal = True
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

    def updateRobotPose (self, odometry):
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

    #using the analytical function generate the
    #pitch yaw and pose of the robot
    def computePosition(self):
        """
        Compute the four parameters to know if the manipulation is possilb
        """
        #Compute the tranformation matrix for the base of the arm
        trans_matrix = tf.transformations.quaternion_matrix(
            [self.robotPose.orientation.x,
             self.robotPose.orientation.y,
             self.robotPose.orientation.z,
             self.robotPose.orientation.w])

        trans_matrix[0, 3] = self.robotPose.position.x
        trans_matrix[1, 3] = self.robotPose.position.y
        trans_matrix[2, 3] = self.robotPose.position.z

        base_arm_auv = np.array([self.base_pose[0],
                                 self.base_pose[1],
                                 self.base_pose[2],
                                 1])

        base_arm_world = np.dot(trans_matrix, base_arm_auv)

        # rospy.loginfo('Base en el Mon ' + str(base_arm_world[0:3]))
        #distance compute euclidean distance
        self.distance_goal = np.sqrt(
            np.power(base_arm_world[0] - self.goalPose.position.x, 2) +
            np.power(base_arm_world[1] - self.goalPose.position.y, 2) +
            np.power(base_arm_world[2] - self.goalPose.position.z, 2)
            )


        #compute the angle alpha and beta
        # we need the position of the base from the valve position
        trans_matrix_valve = tf.transformations.quaternion_matrix(
            [self.goalPose.orientation.x,
             self.goalPose.orientation.y,
             self.goalPose.orientation.z,
             self.goalPose.orientation.w])

        trans_matrix_valve[0, 3] = self.goalPose.position.x
        trans_matrix_valve[1, 3] = self.goalPose.position.y
        trans_matrix_valve[2, 3] = self.goalPose.position.z

        # rospy.loginfo('Panell en el Mon ' + str(self.goalPose.position.x)
        #               + ', ' +  str(self.goalPose.position.y)
        #               + ', ' +  str(self.goalPose.position.z))

        #invert Matrix
        inv_mat = np.zeros([4, 4])
        inv_mat[3, 3] = 1.0
        inv_mat[0:3, 0:3] = np.transpose(trans_matrix_valve[0:3, 0:3])
        inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                 trans_matrix_valve[0:3, 3])
        base_from_valve = np.dot(inv_mat, base_arm_world)

        # rospy.loginfo('base from the valve ' + str(base_from_valve[0:3]))
        self.alpha = np.arctan2( base_from_valve[0], base_from_valve[2] )
        self.beta = np.arctan2( base_from_valve[1], base_from_valve[2] )

        #For the gamma
        trans_matrix_base = tf.transformations.quaternion_matrix(
            [self.robotPose.orientation.x,
             self.robotPose.orientation.y,
             self.robotPose.orientation.z,
             self.robotPose.orientation.w])

        trans_matrix_base[0, 3] = base_arm_world[0]
        trans_matrix_base[1, 3] = base_arm_world[1]
        trans_matrix_base[2, 3] = base_arm_world[2]

        inv_mat_base = np.zeros([4, 4])
        inv_mat_base[3, 3] = 1.0
        inv_mat_base[0:3, 0:3] = np.transpose(trans_matrix_base[0:3, 0:3])
        inv_mat_base[0:3, 3] = np.dot((-1*inv_mat_base[0:3, 0:3]),
                                      trans_matrix_base[0:3, 3])

        valve_pose_world = np.array([self.goalPose.position.x,
                                     self.goalPose.position.y,
                                     self.goalPose.position.z,
                                     1])
        valve_pose_base = np.dot(inv_mat_base, valve_pose_world)

        # rospy.loginfo('valve from the base ' + str(valve_pose_base[0:3]))
        # rospy.loginfo('valve from the base ' + str(valve_pose_world[0:3]))
        self.gamma = np.arctan2( valve_pose_base[1], valve_pose_base[0] )

        # rospy.loginfo('Distance :' + str(self.distance_goal) )
        # rospy.loginfo('Gamma ' + str(self.gamma) )
        # rospy.loginfo('Alpha ' + str(self.alpha) )
        # rospy.loginfo('Beta ' + str(self.beta) )


    def evaluatePosition(self):
        """
        Check if the position computed is in the limits
        """

        #magic stuff with the limtis
        #define manually the sigma and mu
        #possible values mu

        #p(x) =  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)

        if ( (self.distance_goal < self.limit_distance_goal[0] or
              self.distance_goal > self.limit_distance_goal[1] )
             or np.abs(self.gamma) > self.limit_gamma
             or np.abs(self.alpha) > self.limit_alpha
             or (self.beta < self.limit_beta[0] or
              self.beta > self.limit_beta[1] ) ):

            #rospy.loginfo('The robot is outsite of the working area')
            rospy.loginfo('Robot Outsite')
            if (self.distance_goal < self.limit_distance_goal[0] or
                self.distance_goal > self.limit_distance_goal[1] ):
                rospy.loginfo('Distance: ' + str(self.limit_distance_goal[0])
                              + ' < ' + str(self.distance_goal)
                              + ' < ' + str(self.limit_distance_goal[1]))
            if np.abs(self.gamma) > self.limit_gamma :
                rospy.loginfo('Gamma: ' + str(-1.0*self.limit_gamma)
                              + ' < ' + str(self.gamma)
                              + ' < ' + str(self.limit_gamma))
            if np.abs(self.alpha) > self.limit_alpha:
                rospy.loginfo('Alpha: ' + str(-1.0*self.limit_alpha)
                              + ' < ' + str(self.alpha)
                              + ' < ' + str(self.limit_alpha))
            if (self.beta < self.limit_beta[0] or
              self.beta > self.limit_beta[1]):
                rospy.loginfo('Beta: ' + str(self.limit_beta[0])
                              + ' < ' + str(self.beta)
                              + ' < ' + str(self.limit_beta[1]))
            return -1.0
        else :
            #rospy.loginfo('The robot is insite the working area')
            return 1.0


    def run(self):
        rate = rospy.Rate(1.0/self.period)
        is_insite = False
        restart = False
        counter_times = 0
        while not rospy.is_shutdown():
            if self.initRobotPose and self.initGoal :
                self.computePosition()
                safe_rate = self.evaluatePosition()
                #publish the rate to modify the behaviour of the learning program
                if is_insite and not restart:
                    if safe_rate == -1.0:
                        counter_times += 1
                    else:
                        counter_times = 0
                        self.pub_decision.publish(1.0)
                    if (counter_times >= self.limit_wait_outsite and
                        counter_times < self.limit_outsite):
                        rospy.loginfo('Waiting to recover pose')
                        self.pub_decision.publish(0.0)
                    elif counter_times >= self.limit_outsite:
                        self.pub_decision.publish(-1.0)
                        rospy.loginfo('Restart Trajectory')
                        counter_times = 0
                        restart = True
                        is_insite = False
                elif restart:
                    self.pub_decision.publish(-1.0)
                    rospy.loginfo('Restarting')
                    counter_times += 1
                    if counter_times >= self.limit_outsite:
                        restart = False
                else:
                    rospy.loginfo('Waiting to be insite')
                    self.pub_decision.publish(1.0)
                    if safe_rate == 1.0:
                        counter_times += 1
                    else:
                        counter_times = 0
                    if counter_times >= self.limit_enter_zone:
                        is_insite = True
            else:
                rospy.loginfo('Waiting to initialize all the positions')
            rate.sleep()


if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora_misc", "work_area_v2.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate work_area_v2.yaml")

        rospy.init_node('work_area_v2')
        work_area_controller = WorkAreaController( rospy.get_name() )
        work_area_controller.run()
    except rospy.ROSInterruptException: pass
