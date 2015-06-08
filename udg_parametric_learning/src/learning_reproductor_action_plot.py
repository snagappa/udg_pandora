#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_parametric_learning')
import rospy

#use to load the configuration function
from cola2_lib import cola2_ros_lib

import actionlib

#use to normalize the angle
from cola2_lib import cola2_lib


# import the service to call the service
# Warnning I don't know if is needed may be can be seen directly
#from cola2_control.srv import MoveArmTo
# import the message to know the position
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped

# include the message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark

#include message of the point
#from sensor_msgs.msg import Joy

#include the message to send velocities to the robot
from auv_msgs.msg import BodyVelocityReq

import math
import numpy as np
#from scipy import interpolate

#To enable or disable the service
from std_srvs.srv import Empty, EmptyResponse
from cola2_control.srv import JointPose #EFPose
from cola2_control.srv import TurnDesiredDegrees, PushWithAUV
#from udg_parametric_learning.srv import StareLandmark
from udg_parametric_learning.srv import StaticCurrent
#, StaticCurrentResponse, StaticCurrentRequest

from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped, TwistStamped
from learning_pandora.msg import ValveTurningAction, ValveTurningFeedback
from learning_pandora.msg import ValveTurningResult
from learning_pandora.msg import rfdm_msg

#from udg_pandora.srv import WorkAreaError

import threading
import tf

from tf.transformations import euler_from_quaternion

#from learning_dmp_reproductor import LearningDmpReproductor
from learning_dmp_param_reproductor import LearningDmpParamReproductor

from cola2_sim.msg import CsipE5ArmState
#from cola2_arm_dev.msg import CsipE5ArmState

import matplotlib.pyplot as plt

#import warnings

#value to show all the numbers in a matrix
# numpy
# .set_printoptions(threshold=100000)

class learningReproductorAct:
    """
    This class reproduce a trajectory learned with the AUV and the Manipulator
    to turn the valve. The algorithm is switched on using the action library.
    @author: Arnau Carrera
    """
    def __init__(self, name):
        """
        This function initialize the node, load the configuration, initialize
        the Publisher and subscriber and the services. Also open the csv export
        files removing the previous existing.
        """
        self.name = name
        self.get_config()
        #self.getLearnedParameters()
        self.goalPose = Pose()
        self.robotPose = Pose()
        self.valve_orientation = Pose()
        self.armPose = np.zeros(3)
        self.prevPos = np.zeros(self.nbVar)
        self.prevTimeArm = 0.0
        self.currTimeArm = 0.0
        self.prevTimeAUV = 0.0
        self.currTimeAUV = 0.0
        self.currPos = np.zeros(self.nbVar)
        self.currVel = np.zeros(self.nbVar)
        self.currAcc = np.zeros(self.nbVar)
        self.desAcc = np.zeros(self.nbVar)
        self.dataReceived = 0
        self.dataReceivedArm = 0
        self.limit_reach = False
        self.limit_reach_index = 0
        self.dataGoalReceived = False
        self.dataGoalPoseReceived = False
        self.dataGoalOriReceived = False
        self.dataRobotReceived = False
        self.dataComputed = 0
        #Simulation parameter
        self.currPosSim = np.zeros(self.nbVar)
        self.currPosSim[0] = 1.0
        self.currPosSim[1] = 0.0
        self.currPosSim[2] = 2.0
        self.currPosSim[3] = 0.2
        self.currPosSim[4] = 0.05
        self.currPosSim[5] = -0.3
        self.currPosSim[6] = 0.7

        #Work Around to be removed
        self.desPos = np.zeros(self.nbVar)
        self.desVel = np.zeros(self.nbVar)

        self.force_vector = np.zeros(6)
        self.force_vector_old = np.zeros(6)
        self.force_big_update = 0
        self.force_new_data = False

        self.currNbDataRepro = 0
        self.enabled = False
        self.action_in_process = False
        self.initial_s = self.s
        self.action = 1.0

        self.valveOri = 0.0
        self.valveOriInit = False

        #finish condition
        self.h_value = 0.0

        #reverse time value
        self.tf = 0.0
        self.backward = False

        self.param = 0.0

        if self.simulation:
            self.file_export = open(self.exportFile+'_sim.csv', 'w')
        else:
            self.file_export = open(self.exportFile+'_real.csv', 'w')

        self.lock = threading.Lock()
        self.lock_force = threading.Lock()
        self.pub_auv_vel = rospy.Publisher(
            "/cola2_control/body_velocity_req", BodyVelocityReq)
        self.pub_arm_command = rospy.Publisher(
            "/cola2_control/joystick_arm_ef_vel", Joy)
        self.pub_auv_finish = rospy.Publisher(
            "learning/auv_finish", Bool)
        self.pub_arm_des_pose = rospy.Publisher(
            "learning/end_effector_desired_pose", PoseStamped)

        rospy.Subscriber('/pose_ekf_slam/map',
                         Map, self.updateGoalOri,
                         queue_size = 1)
        self.sub_valve = rospy.Subscriber(('/valve_tracker/valve'+
                                           str(self.goal_valve)),
                                          PoseWithCovarianceStamped,
                                          self.updateGoalPose,
                                          queue_size = 1)
        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry,
                         self.updateRobotPose,
                         queue_size = 1)
        rospy.Subscriber('/arm/pose_stamped',
                         PoseStamped,
                         self.updateArmPosition,
                         queue_size = 1)
        rospy.Subscriber('/rfdm/reactive',
                         rfdm_msg,
                         self.updateSafety,
                         queue_size = 1)
        rospy.Subscriber('/current_estimation/current_vector',
                         TwistStamped,
                         self.update_current,
                         queue_size = 1)

        rospy.Subscriber('/csip_e5_arm/arm_state',
                         CsipE5ArmState,
                         self.update_arm_state,
                         queue_size = 1)

        rospy.loginfo('Configuration ' + str(name) + ' Loaded ')

        self.enable_srv = rospy.Service(
            '/learning/enable_reproductor_complete',
            Empty,
            self.enable_fun_srv)

        self.disable_srv = rospy.Service(
            '/learning/disable_reproductor_complete',
            Empty,
            self.disable_fun_srv)

        if self.force_torque_enable:
            rospy.loginfo('Force Torque Enabled ')
            rospy.Subscriber('/force_torque_controller/wrench_stamped',
                             WrenchStamped,
                             self.updateForceTorque,
                             queue_size = 1)

        # Create the action server to attend
        self.valve_turning_action = actionlib.SimpleActionServer(
            '/learning/valve_turning_action',
            ValveTurningAction,
            self.valve_turning_act, False)

        self.valve_turning_action.start()

    def get_config(self):
        """
        This function loads all the parameter form the rosparam server using the
        function developed in the cola2_ros_lib.
        """
        param_dict = {'reproductor_parameters_long': 'learning/reproductor/complete/parameters_long',
                      'reproductor_parameters_short': 'learning/reproductor/complete/parameters_short',
                      'alpha': 'learning/reproductor/complete/alpha',
                      's': 'learning/reproductor/complete/s',
                      'nbVar': 'learning/reproductor/complete/nbVar',
                      'interval_time': 'learning/reproductor/complete/interval_time',
                      'landmark_id': 'learning/reproductor/complete/landmark_id',
                      'simulation': 'learning/reproductor/complete/simulation',
                      'nbDataRepro': 'learning/reproductor/complete/nbDataRepro',
                      'exportFile': 'learning/reproductor/complete/exportFile',
                      'frame_id_goal': 'learning/reproductor/complete/frame_id_goal',
                      'goal_valve': 'learning/reproductor/complete/goal_valve',
                      'learning_param_id': 'learning/reproductor/complete/learning_param_id',
                      'base_pose': '/arm_controller/base_pose',
                      'base_ori': '/arm_controller/base_ori',
                      'force_torque_enable': '/learning/reproductor/complete/force_torque_enable'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.reproductor_parameters = self.reproductor_parameters_short
        rospy.loginfo('Value parameters ' + str(self.reproductor_parameters))

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
                            [pose_msg.pose.pose.orientation.x,
                             pose_msg.pose.pose.orientation.y,
                             pose_msg.pose.pose.orientation.z,
                             pose_msg.pose.pose.orientation.w])[2]
            self.valve_orientation.orientation = pose_msg.pose.pose.orientation
            if not self.dataGoalPoseReceived:
                self.dataGoalPoseReceived = True
                if (self.dataGoalOriReceived and
                    not self.dataGoalReceived):
                    self.dataGoalReceived = True
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
                    if not self.dataGoalOriReceived:
                        self.dataGoalOriReceived = True
                        if (self.dataGoalPoseReceived and
                            not self.dataGoalReceived):
                            self.dataGoalReceived = True
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
            if not self.dataRobotReceived:
                rospy.loginfo('Odometry Initialised')
                robotOriV2 = tf.transformations.quaternion_matrix(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])
                self.dataRobotReceived = True
            if self.dataGoalReceived:
                trans_mat = tf.transformations.quaternion_matrix(
                    [self.goalPose.orientation.x,
                     self.goalPose.orientation.y,
                     self.goalPose.orientation.z,
                     self.goalPose.orientation.w])
                robotOriV2 = tf.transformations.quaternion_matrix(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])

                trans_mat[0, 3] = self.goalPose.position.x
                trans_mat[1, 3] = self.goalPose.position.y
                trans_mat[2, 3] = self.goalPose.position.z

                #invert Matrix
                inv_mat = np.zeros([4, 4])
                inv_mat[3, 3] = 1.0
                inv_mat[0:3, 0:3] = np.transpose(trans_mat[0:3, 0:3])
                inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                         trans_mat[0:3, 3])

                robotPose = np.array(
                    [self.robotPose.position.x,
                     self.robotPose.position.y,
                     self.robotPose.position.z,
                     1])

                robotTrans = np.dot(inv_mat, robotPose)

                self.prevPos[0:4] = self.currPos[0:4]
                self.currPos[0:3] = robotTrans[0:3]
                #Yaw
                goalYaw = euler_from_quaternion(
                    [self.goalPose.orientation.x,
                     self.goalPose.orientation.y,
                     self.goalPose.orientation.z,
                     self.goalPose.orientation.w])[1]
                robotYaw = euler_from_quaternion(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])[2]

                # WORK AROUND CAUTION
                #Same orientation like the AUV, Z down X backward Y lateral
                rot_test = tf.transformations.euler_matrix(0,np.pi/2.0,-np.pi/2.0)

                #new_panel = np.dot(trans_mat[0:3, 0:3], rot_test[0:3, 0:3])
                new_panel = np.dot(trans_mat, rot_test)

                inv_new_panel = np.zeros([4, 4])
                inv_new_panel[3, 3] = 1.0
                inv_new_panel[0:3, 0:3] = np.transpose(new_panel[0:3, 0:3])
                inv_new_panel[0:3, 3] = np.dot((-1*inv_new_panel[0:3, 0:3]),
                                         new_panel[0:3, 3])

                dif_ori = np.dot(robotOriV2[0:3, 0:3],inv_new_panel[0:3,0:3])

                self.currPos[3] = tf.transformations.euler_from_matrix(
                    dif_ori)[2]

                if self.dataReceived == 0:
                    self.currTime = (odometry.header.stamp.secs +
                                     (odometry.header.stamp.nsecs*1E-9))
                    self.dataReceived += 1

                elif self.dataReceived == 1:
                    self.prevPos[0:4] = self.currPos[0:4]
                    self.prevTimeAUV = self.currTimeAUV
                    self.currTimeAUV = (odometry.header.stamp.secs +
                                        (odometry.header.stamp.nsecs*1E-9))
                    self.currVel[0:4] = ((self.currPos[0:4] - self.prevPos[0:4])
                                         /(self.currTimeAUV - self.prevTimeAUV))
                    self.dataReceived += 1
                else:
                    self.prevPos[0:4] = self.currPos[0:4]
                    self.prevTimeAUV = self.currTimeAUV
                    self.currTimeAUV = (odometry.header.stamp.secs +
                                        (odometry.header.stamp.nsecs*1E-9))
                    self.currVel[0:4] = ((self.currPos[0:4] - self.prevPos[0:4])
                                         /(self.currTimeAUV - self.prevTimeAUV))
            else:
                rospy.loginfo(
                    'Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()

    def updateArmPosition(self, data):
        """
        This method update the position of the end-effector. Using the arm
        position published by the arm_controller and the pose of the AUV and
        the distance between the AUV and the base of the arm.
        @param data: Contains the odometry computed in the pose_ekf
        @type data: Odometry message
        """
        self.lock.acquire()
        try:
            if self.dataGoalReceived:
                endeffectorPose = np.array([data.pose.position.x,
                                            data.pose.position.y,
                                            data.pose.position.z,
                                            1])
                self.armOrientation = euler_from_quaternion([
                    data.pose.orientation.x,
                    data.pose.orientation.y,
                    data.pose.orientation.z,
                    data.pose.orientation.w])

                trans_matrix_v2 = tf.transformations.quaternion_matrix(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])

                trans_matrix_v2[0, 3] = self.robotPose.position.x
                trans_matrix_v2[1, 3] = self.robotPose.position.y
                trans_matrix_v2[2, 3] = self.robotPose.position.z

                trans_matrix = tf.transformations.quaternion_matrix(
                    [self.goalPose.orientation.x,
                     self.goalPose.orientation.y,
                     self.goalPose.orientation.z,
                     self.goalPose.orientation.w])

                trans_matrix[0, 3] = self.goalPose.position.x
                trans_matrix[1, 3] = self.goalPose.position.y
                trans_matrix[2, 3] = self.goalPose.position.z

                inv_mat = np.zeros([4, 4])
                inv_mat[3, 3] = 1.0
                inv_mat[0:3, 0:3] = np.transpose(trans_matrix[0:3, 0:3])
                inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                         trans_matrix[0:3, 3])

                robot_base = tf.transformations.euler_matrix(
                    self.base_ori[0],
                    self.base_ori[1],
                    self.base_ori[2])

                robot_base[0, 3] = self.base_pose[0]
                robot_base[1, 3] = self.base_pose[1]
                robot_base[2, 3] = self.base_pose[2]

                arm_base = np.dot(robot_base, endeffectorPose)

                arm_world_pose = np.dot(trans_matrix_v2, arm_base)

                endEfWorld = np.dot(inv_mat, arm_world_pose)

                self.armPose[0:3] = endEfWorld[0:3]
                self.prevPos[4:10] = self.currPos[4:10]
                self.currPos[4:7] = self.armPose

                # Compute orientation

                ori_valve_n = tf.transformations.quaternion_matrix([
                    self.valve_orientation.orientation.x,
                    self.valve_orientation.orientation.y,
                    self.valve_orientation.orientation.z,
                    self.valve_orientation.orientation.w])

                #Same orientation like the AUV, Z down X backward Y lateral
                rot_test = tf.transformations.euler_matrix(np.pi,0.0,0.0)

                #new_panel = np.dot(trans_matrix[0:3, 0:3], rot_test[0:3, 0:3])
                valve_orientated_as_end_effector = np.dot(ori_valve_n, rot_test)

                end_effector_ori = tf.transformations.quaternion_matrix([
                    data.pose.orientation.x,
                    data.pose.orientation.y,
                    data.pose.orientation.z,
                    data.pose.orientation.w])

                ee_ori_base = np.dot(robot_base[0:3, 0:3],
                                     end_effector_ori[0:3,0:3])

                ee_ori_world = np.dot(trans_matrix_v2[0:3, 0:3], ee_ori_base)

                end_effector_ori_frame_valve = np.dot(
                    np.transpose(ee_ori_world),
                    valve_orientated_as_end_effector[0:3, 0:3])

                ee_euler= tf.transformations.euler_from_matrix(
                    end_effector_ori_frame_valve)

                self.armOrientation = ee_euler
                self.currPos[7:10] = self.armOrientation

                if self.dataReceivedArm == 0:
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.dataReceivedArm += 1
                elif self.dataReceivedArm == 1:
                    self.prevPos[4:10] = self.currPos[4:10]
                    self.prevTimeArm = self.currTimeArm
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.currVel[4:10] = (
                        (self.currPos[4:10]-self.prevPos[4:10]) /
                        (self.currTimeArm-self.prevTimeArm))
                    self.dataReceivedArm += 1
                else:
                    self.prevPos[4:10] = self.currPos[4:10]
                    self.prevTimeArm = self.currTimeArm
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.currVel[4:10] = (
                        (self.currPos[4:10]-self.prevPos[4:10]) /
                        (self.currTimeArm-self.prevTimeArm))
            else:
                rospy.loginfo('Goal pose Not initialized')
        finally:
            self.lock.release()

    def updateSafety(self, rfdm_msg):
        '''
        Recive the action value which can make the time move forward or backward
        @param rfdm_msg: Contain a int with the Rfdm_msg
        @type rfdm_msg: Rfdm_msg only contain a float from -1.0 to 1.0
        '''
        self.lock.acquire()
        try:
            self.action = rfdm_msg.reactive_data
        finally:
            self.lock.release()

    def updateForceTorque(self, wrench_msg):
        '''
        Receive the values of the Force in the end_effector
        @param wrench_msg: Contains the force and torque
        @type wrench_msg: Wrench_msg from Geometry
        '''
        self.lock_force.acquire()
        try:
            self.force_vector_old = np.copy(self.force_vector)
            self.force_vector[0] = wrench_msg.wrench.force.x
            self.force_vector[1] = wrench_msg.wrench.force.y
            self.force_vector[2] = wrench_msg.wrench.force.z
            self.force_vector[3] = wrench_msg.wrench.torque.x
            self.force_vector[4] = wrench_msg.wrench.torque.y
            self.force_vector[5] = wrench_msg.wrench.torque.z
        finally:
            self.lock_force.release()

    def update_current(self, twist_msg):
        '''
        Receive the twist stamped from the Current estimator. Convert this
        vector to a particular values.
        '''
        x = twist_msg.twist.linear.x
        y = twist_msg.twist.linear.y
        z = twist_msg.twist.linear.z

        yaw = twist_msg.angular.z

        self.param = np.linalg.norm([x,y,z])

    def update_arm_state(self, arm_state_msg):
        '''
        Receive the arm_state and check if some of the 3 first joints are in the
        limit, the roll is excluded.
        '''
        if any(arm_state_msg.joint_max_limit[0:3]) or any(arm_state_msg.joint_min_limit[0:3]):
            #rospy.loginfo('Some limit')
            if any(arm_state_msg.joint_max_limit[0:3]):
                self.limit_reach_index = arm_state_msg.joint_max_limit[0:3].index(True)+1
            else: # any(arm_state_msg.joint_min_limit[0:3]):
                self.limit_reach_index = -1*(arm_state_msg.joint_min_limit[0:3].index(True)+1)
            self.limit_reach = True
        else:
            self.limit_reach = False

    def load_trajectory(self, file_name, samples):
        """
        Load Trajectory from the last point to the beginning
        """
        print 'Loading Trajectory ' + file_name + ' ' +  str(samples) + ' :'
        demonstrations = []
        if len(samples) != 0:
            for n in xrange(len(samples)):
            #print 'Loading Demonstration ' + file_name + "_" + str(ni)
                ni = samples[n]
                if type(file_name) is str:
                    logfile = open(file_name + "_" + str(ni) + ".csv",
                                   "r").readlines()
                else:
                #The file name is a list of elements
                    logfile = open(file_name[n] + "_" + str(ni) + ".csv",
                                   "r").readlines()
                # vars = np.zeros((1, self.nbVar))
                # Added the time to the var
                data_demo = np.array([[]])
                for line in logfile:
                    if len(data_demo[0]) == 0:
                        data_demo = np.array([line.split()], dtype=np.float64)
                    else:
                        data_demo = np.append(
                            data_demo,
                            np.array([line.split()], dtype=np.float64),
                            axis=0)
                demonstrations.append(data_demo)
        else:
            logfile = open(file_name + ".csv", "r").readlines()
            data_demo = np.array([[]])
            for line in logfile:
                if len(data_demo[0]) == 0:
                    data_demo = np.array([line.split()], dtype=np.float64)
                else:
                    data_demo = np.append(
                        data_demo,
                        np.array([line.split()], dtype=np.float64),
                        axis=0)
            demonstrations.append(data_demo)
        return demonstrations


    def enable_fun_srv(self, req):
        '''
        Service to enable the action when there is not other action in progress
        At the beginning the static current service is asked.
        @param req: Empty parameter
        @type req: Empty
        '''
        if not self.action_in_process :
            #TODO Uncomment this lines
            # if not self.simulation:
            #     rospy.wait_for_service('/current_estimator/static_estimation')
            #     static_estimation_srv = rospy.ServiceProxy(
            #         '/current_estimator/static_estimation',
            #         StaticCurrent)
            #     response = static_estimation_srv.call()

            #     x = response.current_estimation[0]
            #     y = response.current_estimation[1]
            #     z = response.current_estimation[2]
            # #self.param = np.linalg.norm([x,y,z])
            #     self.param = y

            self.param = 1.0

            self.enabled = True
            rospy.loginfo('%s Enabled', self.name)
        else:
            rospy.loginfo('The %s is doing an action', self.name)
        return EmptyResponse()

    def disable_fun_srv(self, req):
        '''
        Service to disable the action when there is not other action in progress
        @param req: Empty parameter
        @type req: Empty
        '''
        if not self.action_in_process :
            self.enabled = False
            self.s = self.initial_s
            self.action = 1.0
            rospy.loginfo('%s Disabled', self.name)
        else:
            rospy.loginfo('The %s is doing an action', self.name)
        return EmptyResponse()

    def play(self):
        '''
        Main function which runs the main operations except for the actions
        which are executed in its own function.
        '''


#        pub = rospy.Publisher('arm', )
        rate = rospy.Rate(1.0/self.interval_time)
        #Find the parameters to load
        #TODO Fix what hapend with the action
        path = roslib.packages.get_pkg_subdir('udg_parametric_learning',
                                              'parametric_data',
                                              False)
        # choose the file of the list in the learning directory
        # build the path and the file name
        nb_groups = 2
        #file_path = path + '/' + self.reproductor_parameters[0]
        dmp_z = LearningDmpParamReproductor(
            self.reproductor_parameters[0],
            path,
            #self.nbVar,
            1,
            self.alpha,
            self.interval_time,
            nb_groups)
        #rospy.loginfo('Loadded file ' + str())
        #file_path = path + '/' + self.reproductor_parameters[1]
        dmp_x_y_yaw = LearningDmpParamReproductor(
            self.reproductor_parameters[1],
            path,
            #self.nbVar,
            3,
            #1,
            self.alpha,
            self.interval_time,
            nb_groups)

        # ARM
        dmp_arm_z = LearningDmpParamReproductor(
            self.reproductor_parameters[2],
            path,
            1,
            self.alpha,
            self.interval_time,
            nb_groups)
        dmp_arm_x_y_yaw = LearningDmpParamReproductor(
            self.reproductor_parameters[3],
            path,
            3,
            #1,
            self.alpha,
            self.interval_time,
            nb_groups)

        time = 0
        # Ask for the parameter value

        while not rospy.is_shutdown():
            self.lock.acquire()
            #rospy.loginfo('Num states ' + str(self.nbSates) + ' Llegits ' + str(self.))
            try:
                # rospy.loginfo('Curr Pos Rob ' + str(self.currPos[0])
                #               + ', ' + str(self.currPos[1])
                #               + ', ' + str(self.currPos[2]))
                # rospy.loginfo('Curr Pos Arm ' + str(self.currPos[4])
                #               + ', ' + str(self.currPos[5])
                #               + ', ' + str(self.currPos[6]))
                if self.enabled:
                    if not self.simulation:
                        if self.dataReceived > 1 and self.dataReceivedArm > 1:
                            #rospy.loginfo('Current Alignment ' + str(self.currPos[9]) )
                            [des_pose_z, des_vel_z] = dmp_z.generateNewPose(
                                self.currPos, self.currVel,
                                self.action, self.param)
                            [des_pose_x_y_yaw, des_vel_x_y_yaw] = dmp_x_y_yaw.generateNewPose(
                                self.currPos, self.currVel,
                                self.action, self.param)
                            [des_pose_arm_z, des_vel_arm_z] = dmp_arm_z.generateNewPose(
                                self.currPos, self.currVel,
                                self.action, self.param)
                            [des_pose_arm_x_y_yaw, des_vel_arm_x_y_yaw] = dmp_arm_x_y_yaw.generateNewPose(
                                self.currPos, self.currVel,
                                self.action, self.param)
                            if (len(des_pose_z) != 0 and len(des_pose_x_y_yaw) != 0
                                and len(des_pose_arm_z) != 0
                                and len(des_pose_arm_x_y_yaw) != 0):
                                self.desPos[0:2] = des_pose_x_y_yaw[0:2]
                                self.desVel[0:2] = des_vel_x_y_yaw[0:2]
                                self.desPos[1] = des_pose_x_y_yaw[1]
                                self.desVel[1] = des_vel_x_y_yaw[1]
                                self.desPos[2] = des_pose_z
                                self.desVel[2] = des_vel_z
                                self.desPos[3] = des_pose_x_y_yaw[2]
                                self.desVel[3] = des_vel_x_y_yaw[2]
                                # ARM
                                self.desPos[4:6] = des_pose_arm_x_y_yaw[0:2]
                                self.desVel[4:6] = des_vel_arm_x_y_yaw[0:2]
                                self.desPos[6] = des_pose_arm_z
                                self.desVel[6] = des_vel_arm_z
                                self.desPos[9] = des_pose_arm_x_y_yaw[2]
                                self.desVel[9] = des_vel_arm_x_y_yaw[2]
                                #Desired Pose Reza, Removed
                                # desPose_msg = PoseStamped()
                                # desPose_msg.header.stamp = rospy.get_rostime()
                                # desPose_msg.header.frame_id = "valve2"
                                # desPose_msg.pose.position.x = self.desPos[0] #des_pose[0]
                                # desPose_msg.pose.position.y = self.desPos[1] #des_pose[1]
                                # desPose_msg.pose.position.z = self.desPos[2] #des_pose[2]
                                # desPose_msg.pose.orientation.x = 0
                                # desPose_msg.pose.orientation.y = 0
                                # desPose_msg.pose.orientation.z = 0
                                # desPose_msg.pose.orientation.w = 1
                                # self.pub_arm_des_pose.publish(desPose_msg)
                                self.publishCommands()
                            else:
                                rospy.loginfo('Learning has finished ')
                                #self.lock.release()
                                self.file_export.close()
                                break
                    else:
#                        self.simulatedNewPose()
                        if time == 0 :
                            time = rospy.get_time()
                            self.currPos = self.currPosSim
                        else:
                            time += self.interval_time

                        [des_pose_z, des_vel_z] = dmp_z.generateNewPose(
                            self.currPos, self.currVel, self.action, self.param)
                        # rospy.loginfo('Des_pose z ' + str(des_pose_z))
                        [des_pose_x_y_yaw, des_vel_x_y_yaw] = dmp_x_y_yaw.generateNewPose(
                            self.currPos, self.currVel, self.action, self.param)
                        # rospy.loginfo('Des_pose XYYAW ' + str(des_pose_x_y_yaw))
                        # rospy.loginfo('Curr pose ' + str(self.currPos))
                        [des_pose_arm_z, des_vel_arm_z] = dmp_arm_z.generateNewPose(
                            self.currPos, self.currVel, self.action, self.param)
                        [des_pose_arm_x_y_yaw, des_vel_arm_x_y_yaw] = dmp_arm_x_y_yaw.generateNewPose(
                            self.currPos, self.currVel, self.action, self.param)
                        if (len(des_pose_z) != 0 and len(des_pose_x_y_yaw) != 0
                            and len(des_pose_arm_z) != 0
                            and len(des_pose_arm_x_y_yaw) != 0 ):
                            self.currPos[0:2] = des_pose_x_y_yaw[0:2]
                            self.currVel[0:2] = des_vel_x_y_yaw[0:2]
                            # self.currPos[0] = des_pose_x_y_yaw[0]
                            # self.currVel[0] = des_vel_x_y_yaw[0]
                            # self.currPos[1] = des_pose_x_y_yaw[1]
                            # self.currVel[1] = des_vel_x_y_yaw[1]
                            self.currPos[2] = des_pose_z
                            self.currVel[2] = des_vel_z
                            self.currPos[3] = des_pose_x_y_yaw[2]
                            self.currVel[3] = des_vel_x_y_yaw[2]
                            # ARM
                            self.currPos[4:6] = des_pose_arm_x_y_yaw[0:2]
                            self.currVel[4:6] = des_vel_arm_x_y_yaw[0:2]
                            self.currPos[6] = des_pose_arm_z
                            self.currVel[6] = des_vel_arm_z
                            self.currPos[9] = des_pose_arm_x_y_yaw[2]
                            self.currVel[9] = des_vel_arm_x_y_yaw[2]
                            s = (repr(time) + " " +
                                 repr(self.currPos[0]) + " " +
                                 repr(self.currPos[1]) + " " +
                                 repr(self.currPos[2]) + " " +
                                 repr(self.currPos[3]) + " " +
                                 repr(self.currPos[4]) + " " +
                                 repr(self.currPos[5]) + " " +
                                 repr(self.currPos[6]) + " " +
                                 repr(self.currPos[7]) + " " +
                                 repr(self.currPos[8]) + " " +
                                 repr(self.currPos[9]) + "\n")
                            self.file_export.write(s)
                        else:
                            rospy.loginfo('Learning has finished ')
                            #self.lock.release()
                            self.file_export.close()
                            break
            finally:
                self.lock.release()
            if not self.simulation:
                rate.sleep()
            # rospy.sleep(self.interval_time)

    def valve_turning_act(self, goal):
        """
        This function is call when a action is requested. The action function
        iterates until the DMP finish the execution or is preempted.
        @param goal: This param contains the id of the desired valve
        @type goal: ValveTurningAction
        """
        demos_group_2 = self.load_trajectory(
            '../parametric_data/trajectory_demonstration', [67,69,70])
        plt.ion()
        f_auv, axis_auv = plt.subplots(4, sharex=True)
        f_auv.suptitle("AUV")
        for i in xrange(len(demos_group_2)):
            axis_auv[0].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,1], color='r')
            axis_auv[1].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,2], color='r')
            axis_auv[2].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,3], color='r')
            axis_auv[3].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,4], color='r')
        line_x_auv, = axis_auv[0].plot([], [], color='b')
        line_y_auv, = axis_auv[1].plot([], [], color='b')
        line_z_auv, = axis_auv[2].plot([], [], color='b')
        line_yaw_auv, = axis_auv[3].plot([], [], color='b')
 
        #plt.show()
        plt.draw()
        f_ee, axis_ee = plt.subplots(4, sharex=True)
        f_ee.suptitle("End-effector")
        for i in xrange(len(demos_group_2)):
            axis_ee[0].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,5], color='r')
            axis_ee[1].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,6], color='r')
            axis_ee[2].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,7], color='r')
            axis_ee[3].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,10], color='r')

        line_x_ee, = axis_ee[0].plot([], [], color='b')
        line_y_ee, = axis_ee[1].plot([], [], color='b')
        line_z_ee, = axis_ee[2].plot([], [], color='b')
        line_roll_ee, = axis_ee[3].plot([], [], color='b')
        
        #plt.show()
        plt.draw()
        #plt.ioff()

        self.goal_valve = goal.valve_id
        self.sub_valve.unregister()
        self.sub_valve = rospy.Subscriber(('/valve_tracker/valve'+
                                           str(self.goal_valve)),
                                          PoseWithCovarianceStamped,
                                          self.updateGoalPose,
                                          queue_size = 1)

        #Set the id of the file which will be learned
        if goal.long_approach == True:
            self.reproductor_parameters = self.reproductor_parameters_long
        elif goal.long_approach == False:
            self.reproductor_parameters = self.reproductor_parameters_short

        rospy.loginfo('Start Action enable')

        #Load the learned data for the desired behaviour
        path = roslib.packages.get_pkg_subdir('udg_parametric_learning',
                                              'parametric_data',
                                              False)
        # choose the file of the list in the learning directory
        # build the path and the file name
        #file_path = path + '/' + self.reproductor_parameters[0]
        nb_groups = 2
        dmp_z = LearningDmpParamReproductor(
            self.reproductor_parameters[0],
            path,
            1,
            self.alpha,
            self.interval_time,
            nb_groups)
        #file_path = path + '/' + self.reproductor_parameters[1]
        dmp_x_y_yaw = LearningDmpParamReproductor(
            self.reproductor_parameters[1],
            path,
            3,
            self.alpha,
            self.interval_time,
            nb_groups)
        dmp_arm_z = LearningDmpParamReproductor(
            self.reproductor_parameters[2],
            path,
            1,
            self.alpha,
            self.interval_time,
            nb_groups)
        #file_path = path + '/' + self.reproductor_parameters[3]
        dmp_arm_x_y_yaw = LearningDmpParamReproductor(
            self.reproductor_parameters[3],
            path,
            3,
            self.alpha,
            self.interval_time,
            nb_groups)

        #Restartin position data aboid multiple loop while waiting to call the action
        self.dataReceivedArm = 0
        self.dataReceived = 0

        # Asking for the estimation of the current before starting
        rospy.loginfo('Asking for the Current Estimation')
        rospy.wait_for_service('/current_estimator/static_estimation')
        static_estimation_srv = rospy.ServiceProxy(
            '/current_estimator/static_estimation',
            StaticCurrent)
        response = static_estimation_srv.call()

        x = response.current_estimation[0]
        y = response.current_estimation[1]
        z = response.current_estimation[2]
        self.param = np.linalg.norm([x,y,z])
        # FAST PARAM
        self.param = 1.0

        rospy.loginfo('Disabling valve update')
        rospy.wait_for_service('/valve_tracker/disable_update_valve_orientation')
        disable_valve_update_srv = rospy.ServiceProxy(
            '/valve_tracker/disable_update_valve_orientation',
            Empty)
        disable_valve_update_srv.call()

        rospy.loginfo('Starting Valve turning task')

        rate = rospy.Rate(1.0/self.interval_time)
        success = False
        preempted = False
        self.enabled = False
        self.action_in_process = True
        push_srv = rospy.ServiceProxy('/cola2_control/push_desired_froce',
                                      PushWithAUV)
        plot_init_time = rospy.get_time()
        count_plot = 0
        while not success and not preempted: #and self.force_big_update == 0:
            if self.dataReceived > 1 and self.dataReceivedArm > 1:
                if not self.simulation:
                    #success = self.generateNewPose()
                    #rospy.loginfo('Current Alignment ' + str(self.currPos[9]) )
                    [des_pose_z, des_vel_z] = dmp_z.generateNewPose(
                        self.currPos, self.currVel, self.action, self.param)
                    [des_pose_x_y_yaw, des_vel_x_y_yaw] = dmp_x_y_yaw.generateNewPose(
                        self.currPos, self.currVel, self.action, self.param)
                    [des_pose_arm_z, des_vel_arm_z] = dmp_arm_z.generateNewPose(
                        self.currPos, self.currVel, self.action, self.param)
                    [des_pose_arm_x_y_yaw, des_vel_arm_x_y_yaw] = dmp_arm_x_y_yaw.generateNewPose(
                        self.currPos, self.currVel, self.action, self.param)
                    if (len(des_pose_z) != 0 and len(des_pose_x_y_yaw) != 0
                        and len(des_pose_arm_z) != 0
                        and len(des_pose_arm_x_y_yaw) != 0 ):
                        # self.desPos[0:4] = des_pose[0:4]
                        # self.desVel[0:4] = des_vel[0:4]
                        self.desPos[0:2] = des_pose_x_y_yaw[0:2]
                        self.desVel[0:2] = des_vel_x_y_yaw[0:2]
                        self.desPos[2] = des_pose_z
                        self.desVel[2] = des_vel_z
                        self.desPos[3] = des_pose_x_y_yaw[2]
                        self.desVel[3] = des_vel_x_y_yaw[2]
                        # ARM
                        # Now the difference is performed outsite instead of in
                        # the pulihsCommands function
                        self.desPos[4:6] = des_pose_arm_x_y_yaw[0:2]
                        self.desVel[4:6] = (des_vel_arm_x_y_yaw[0:2]
                                            - des_vel_x_y_yaw[0:2])
                        self.desPos[6] = des_pose_arm_z
                        self.desVel[6] = des_vel_arm_z -des_vel_z
                        self.desPos[9] = des_pose_arm_x_y_yaw[2]
                        self.desVel[9] = des_vel_arm_x_y_yaw[2]
                        desPose_msg = PoseStamped()
                        desPose_msg.header.stamp = rospy.get_rostime()
                        desPose_msg.header.frame_id = "valve2"
                        desPose_msg.pose.position.x = self.desPos[0] #des_pose[0]
                        desPose_msg.pose.position.y = self.desPos[1] #des_pose[1]
                        desPose_msg.pose.position.z = self.desPos[2] #des_pose[2]
                        desPose_msg.pose.orientation.x = 0
                        desPose_msg.pose.orientation.y = 0
                        desPose_msg.pose.orientation.z = 0
                        desPose_msg.pose.orientation.w = 1
                        self.pub_arm_des_pose.publish(desPose_msg)
                        self.publishCommands()
                        # Printing new pose
                        plot_time = rospy.get_time() - plot_init_time
                        line_x_auv.set_xdata(np.append(line_x_auv.get_xdata(), plot_time))
                        line_x_auv.set_ydata(np.append(line_x_auv.get_ydata(), self.currPos[0]))

                        line_y_auv.set_xdata(np.append(line_y_auv.get_xdata(), plot_time))
                        line_y_auv.set_ydata(np.append(line_y_auv.get_ydata(), self.currPos[1]))

                        line_z_auv.set_xdata(np.append(line_z_auv.get_xdata(), plot_time))
                        line_z_auv.set_ydata(np.append(line_z_auv.get_ydata(), self.currPos[2]))

                        line_yaw_auv.set_xdata(np.append(line_yaw_auv.get_xdata(), plot_time))
                        line_yaw_auv.set_ydata(np.append(line_yaw_auv.get_ydata(), self.currPos[3]))

                        line_x_ee.set_xdata(np.append(line_x_ee.get_xdata(), plot_time))
                        line_x_ee.set_ydata(np.append(line_x_ee.get_ydata(), self.currPos[4]))

                        line_y_ee.set_xdata(np.append(line_y_ee.get_xdata(), plot_time))
                        line_y_ee.set_ydata(np.append(line_y_ee.get_ydata(), self.currPos[5]))

                        line_z_ee.set_xdata(np.append(line_z_ee.get_xdata(), plot_time))
                        line_z_ee.set_ydata(np.append(line_z_ee.get_ydata(), self.currPos[6]))

                        line_roll_ee.set_xdata(np.append(line_roll_ee.get_xdata(), plot_time))
                        line_roll_ee.set_ydata(np.append(line_roll_ee.get_ydata(), self.currPos[9]))

                        if count_plot >= 20:
                            rospy.loginfo('Drawing')
                            f_auv.canvas.draw()
                            f_ee.canvas.draw()
                            count_plot = 0
                        else:
                            rospy.loginfo('Counting' + str(count_plot))
                            count_plot = count_plot + 1

                        success = False
                    else:
                        success = True

                    if self.force_torque_enable and success == False:
                        self.lock_force.acquire()
                        try:
                            # rospy.loginfo('Force in Z ' + str(np.abs(self.force_vector[2] - self.force_vector_old[2])) + ' Force ' + str(self.force_vector[2]))
                            # rospy.loginfo('Action value ' + str(self.action))
                            if (np.abs(self.force_vector[2] - self.force_vector_old[2]) >= 1.0 and
                                self.force_vector[2] < -3.0):
                                #self.force_big_update = 1
                                success = True
                        finally:
                            self.lock_force.release()
                else :
                    #success = self.simulatedNewPose()
                    if time == 0 :
                        time = rospy.get_time()
                        self.currPos = self.currPosSim
                    else:
                        time += self.interval_time
                    [des_pose_z, des_vel_z] = dmp_z.generateNewPose(
                        self.currPos, self.currVel, self.action)
                    [des_pose_x_y_yaw, des_vel_x_y_yaw] = dmp_x_y_yaw.generateNewPose(
                        self.currPos, self.currVel, self.action)
                    [des_pose_arm_z, des_vel_arm_z] = dmp_arm_z.generateNewPose(
                        self.currPos, self.currVel, self.action)
                    [des_pose_arm_x_y_yaw, des_vel_arm_x_y_yaw] = dmp_arm_x_y_yaw.generateNewPose(
                        self.currPos, self.currVel, self.action)
                    if (len(des_pose_z) != 0 and len(des_pose_x_y_yaw) != 0
                        and len(des_pose_arm_z) != 0
                        and len(des_pose_arm_x_y_yaw) != 0 ):
                        self.currPos[0:2] = des_pose_x_y_yaw[0:2]
                        self.currVel[0:2] = des_vel_x_y_yaw[0:2]
                        self.currPos[2] = des_pose_z
                        self.currVel[2] = des_vel_z
                        self.currPos[3] = des_pose_x_y_yaw[2]
                        self.currVel[3] = des_vel_x_y_yaw[2]
                        # ARM
                        self.currPos[4:6] = des_pose_arm_x_y_yaw[0:2]
                        self.currVel[4:6] = des_vel_arm_x_y_yaw[0:2]
                        self.currPos[6] = des_pose_arm_z
                        self.currVel[6] = des_vel_arm_z
                        self.currPos[9] = des_pose_arm_x_y_yaw[2]
                        self.currVel[9] = des_vel_arm_x_y_yaw[2]
                        s = (repr(time) + " " +
                             repr(self.currPos[0]) + " " +
                             repr(self.currPos[1]) + " " +
                             repr(self.currPos[2]) + " " +
                             repr(self.currPos[3]) + " " +
                             repr(self.currPos[4]) + " " +
                             repr(self.currPos[5]) + " " +
                             repr(self.currPos[6]) + " " +
                             repr(self.currPos[7]) + " " +
                             repr(self.currPos[8]) + " " +
                             repr(self.currPos[9]) + "\n")
                        self.file_export.write(s)
                        
                        success = False
                    else:
                        rospy.loginfo('Learning has finished ')
                        #self.lock.release()
                        self.file_export.close()
                        success = True
            else:
                rospy.loginfo('Waiting to initialize all the data')
            if self.valve_turning_action.is_preempt_requested():
                rospy.loginfo('%s: Preempted Valve Turning', self.name)
                preempted = True
            else :
                #Create Feedback response
                feedback = ValveTurningFeedback()
                # rospy.loginfo('Values of pose ' + str(self.currPos[4:7]))
                # rospy.loginfo('Sum ' + str(self.currPos[4:7]))
                feedback.dist_endeffector_valve = np.sqrt(
                    np.sum(self.currPos[4:7]**2))
                feedback.time_spend = -math.log(self.s)/self.alpha
                self.valve_turning_action.publish_feedback(feedback)
                #Sleep
                rate.sleep()
        #Finished or aborted
        result = ValveTurningResult()
        plt.ioff()
        if preempted:
            result.valve_turned = False
            self.valve_turning_action.set_preempted()
            #retart the time
            self.s = self.initial_s
        else :
            # Push until it touch the valve
            # Turn the valve and the desired degrees
            # stop pushing and go back
            # fold the arm
            # rospy.wait_for_service('/cola2_control/push_desired_froce')
            # rospy.wait_for_service('/cola2_control/turnDesiredRadians')
            # rospy.wait_for_service('/cola2_control/disable_push')
            try:

                #WORK AROUND: REALLY BIG
                # param equal 1 is 45 and 0 is 0
                # update this with the real orientation between the auv and the panel
                #angle_force = 55.0 * self.param
                #force_x = np.cos(angle_force)*30.0
                #force_y = np.sin(angle_force)*30.0
                #push_srv = push_srv([force_x, force_y, 0.0, 0.0, 0.0, 0.0])
                push_srv = push_srv([5.0, 30.0, 0.0, 0.0, 0.0, 0.0])
                error_code = 0
                rospy.loginfo('Pushing the valve ')
                rospy.sleep(3.0)

                turn_srv = rospy.ServiceProxy('/cola2_control/turnDesiredRadians',
                                               TurnDesiredDegrees)
                rospy.loginfo('desired increment ' + str(goal.desired_increment))
                res = turn_srv(goal.desired_increment)
                if not res.success:
                    error_code = 1
                print 'Res ' + str(res.success)

                # Is needed a slipt to be sure we turn the valve ?
                #rospy.sleep(1.0)

                #stop the push
                dis_push_srv = rospy.ServiceProxy('/cola2_control/disable_push',
                                                  Empty)

                dis_push_srv()
                rospy.loginfo('Stop Pushing and go backwards')
                #go backward
                rate = rospy.Rate(10)
                #rospy.loginfo('Init bucle')
                for i in range(80):
                    #rospy.loginfo('Going backward')
                    vel_com = BodyVelocityReq()
                    vel_com.header.stamp = rospy.get_rostime()
                    vel_com.goal.priority = 10
                    #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
                    vel_com.goal.requester = 'learning_algorithm'
                    vel_com.twist.linear.x = -0.1
                    vel_com.twist.linear.y = 0.0
                    vel_com.twist.linear.z = 0.0
                    vel_com.twist.angular.z = 0.0

                    #disabled_axis boby_velocity_req
                    vel_com.disable_axis.x = False # True False
                    vel_com.disable_axis.y = False # True False
                    vel_com.disable_axis.z = True # True False
                    vel_com.disable_axis.roll = True
                    vel_com.disable_axis.pitch = True
                    vel_com.disable_axis.yaw = False # True False
                    self.pub_auv_vel.publish(vel_com)
                    rate.sleep()

                # Fold the arm
                # fold_arm_srv = rospy.ServiceProxy('/cola2_control/setPoseEF',
                #                                   EFPose)
                # value = fold_arm_srv([0.45, 0.0, 0.11, 0.0, 0.0, 0.0 ])

                fold_arm_srv = rospy.ServiceProxy('/cola2_control/setJointPose',
                                                  JointPose)
                value = fold_arm_srv([0.0, 50.0, -30.0, 0.0, 0.0])

                rospy.loginfo('Enabling valve update')
                rospy.wait_for_service('/valve_tracker/enable_update_valve_orientation')
                enable_valve_update_srv = rospy.ServiceProxy(
                    '/valve_tracker/enable_update_valve_orientation',
                    Empty)
                enable_valve_update_srv.call()


                for i in range(40):
                    #rospy.loginfo('Going backward')
                    # vel_com = BodyVelocityReq()
                    # vel_com.header.stamp = rospy.get_rostime()
                    # vel_com.goal.priority = 10
                    # #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
                    # vel_com.goal.requester = 'learning_algorithm'
                    # vel_com.twist.linear.x = -0.0
                    # vel_com.twist.linear.y = 0.0
                    # vel_com.twist.linear.z = 0.0
                    # vel_com.twist.angular.z = 0.0
                    # #disabled_axis boby_velocity_req
                    # vel_com.disable_axis.x = False # True False
                    # vel_com.disable_axis.y = False # True False
                    # vel_com.disable_axis.z = True # True False
                    # vel_com.disable_axis.roll = True
                    # vel_com.disable_axis.pitch = True
                    # vel_com.disable_axis.yaw = False # True False
                    # self.pub_auv_vel.publish(vel_com)
                    rate.sleep()
                rospy.loginfo('Finish')
                result.valve_turned = res.success
                result.error_code = error_code
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        # JUST FOR TEST
        rospy.loginfo('Finish')
        result.valve_turned = res.success


        self.enabled = False
        self.s = self.initial_s
        self.action = 1.0

        #Stop all the movement sending Zero velocities
        # joy_command = Joy()
        # joy_command.axes.append(0.0)
        # joy_command.axes.append(0.0)
        # joy_command.axes.append(0.0)
        # joy_command.axes.append(0.0)
        # joy_command.axes.append(0.0)
        # joy_command.axes.append(0.0)
        # self.pub_arm_command.publish(joy_command)
        # rospy.loginfo('Joy Message stop sent !!!!!')

        vel_com = BodyVelocityReq()
        vel_com.header.stamp = rospy.get_rostime()
        vel_com.goal.priority = 10
        #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        vel_com.goal.requester = 'learning_algorithm'
        vel_com.twist.linear.x = 0.0
        vel_com.twist.linear.y = 0.0
        vel_com.twist.linear.z = 0.0
        vel_com.twist.angular.z = 0.0

        #disabled_axis boby_velocity_req
        vel_com.disable_axis.x = False # True False
        vel_com.disable_axis.y = False # True False
        vel_com.disable_axis.z = False # True False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = False # True False
        self.pub_auv_vel.publish(vel_com)

        self.action_in_process = False
        self.valve_turning_action.set_succeeded(result)

    def update_plot(self, line, value, time):
        line.set_x_data(np.append(line.get_x_data(), current_time-init_time))
        

    def generateNewPose(self):
        t = -math.log(self.s)/self.alpha
        #self.tf
        # if self.backward :
        #     t = self.tf + math.log(self.s)/self.alpha
        # else :
        #     t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = np.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])


        # normalize the value
        if t > self.Mu_t[self.numStates-1]+(self.Sigma_t[self.numStates-1]*1.2):
            rospy.loginfo('The time used in the demonstration is exhausted')
            self.enabled = False
            self.s = self.initial_s
            self.pub_auv_finish.publish(True)
            return True
        else:
            self.h_value = h[self.numStates-1]
            h = h / np.sum(h)

        #rospy.loginfo('H values ' + str(h.tolist()))

        # if np.sum(h) <= 0.00001:
        #     rospy.loginfo('The time used in the demonstration is exhausted')
        #     self.enabled = False
        #     self.s = self.initial_s
        #     self.pub_auv_finish.publish(True)
        #     return True
        # else:
        #     h = h / np.sum(h)

        #init to vectors
        currTar = np.zeros(self.nbVar)
        currWp = np.zeros(shape=(self.nbVar, self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State


        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #rospy.loginfo('Current Tar '+ str(currTar))
        #rospy.loginfo('Current Wp '+ str(currWp))
        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);


        diff = currTar-self.currPos
        diff[3] = cola2_lib.normalizeAngle(diff[3])
        #TODO: No falta el Roll
        #rospy.loginfo('Kv ' + str(self.kV.tolist()))
        self.desAcc = np.dot(
            currWp, diff) - (self.kV*self.currVel)
        # action is a scalar value to evaluate the safety
        self.desAcc = self.desAcc #* math.fabs(self.action)
        #rospy.loginfo('Des Acc' + str(self.desAcc))

        self.desVel = self.currVel + self.desAcc * self.interval_time
        #NOT needed
        self.desPos = self.currPos + self.desVel * self.interval_time

        # rospy.loginfo('Desired Angle ' + str(self.desPos[3]) +
        #               ' Current Angle ' + str(self.currPos[3]) +
        #               ' Desired Vel ' + str(self.desVel[3]))


        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        #self.s = self.s + (-self.alpha*self.s)*self.interval_time
        self.s = self.s - self.alpha*self.s*self.interval_time*self.action#*1.5

        #rospy.loginfo('Value of S ' + str(self.s))
        if (self.s < 1E-200):
            rospy.loginfo('!!!!!!!! AUV trajectory Finish !!!!!!!!!')
            self.enabled = False
            self.s = self.initial_s
            self.pub_auv_finish.publish(True)
            return True
        return False

    def publishCommands(self):
        ##############################################
        # Compute the AUV velocity
        ##############################################
        vel_panel = np.asarray(
            [self.desVel[0],
             self.desVel[1],
             self.desVel[2],
             0])

        # if self.limit_reach:
        #     vel_panel[0] += self.desVel[4]
        #     vel_panel[1] += self.desVel[5]
        #     vel_panel[2] += self.desVel[6]


        # rospy.loginfo('Current Roll ' + str(self.currPos[9]))
        # rospy.loginfo('Des Roll ' + str(self.desPos[9]))
        # rospy.loginfo('Command ' + str(-2.0*self.desVel[9]))

        # rospy.loginfo('Current Pose ' + str(self.currPos[0])
        #               + ', ' + str(self.currPos[1])
        #               + ', ' + str(self.currPos[2]))

        # rospy.loginfo('Des Pose ' + str(self.desPos[0])
        #               + ', ' + str(self.desPos[1])
        #               + ', ' + str(self.desPos[2]))

        # rospy.loginfo('Des Pose ' + str(self.desPos[0])
        #               + ', ' + str(self.desPos[1])
        #               + ', ' + str(self.desPos[2]))

        # rospy.loginfo('Curr Vel ' + str(self.currVel[0])
        #               + ', ' + str(self.currVel[1])
        #               + ', ' + str(self.currVel[2]))

        # rospy.loginfo('Des Vel ' + str(self.desVel[0])
        #               + ', ' + str(self.desVel[1])
        #               + ', ' + str(self.desVel[2]))

        vel_panel_ee = np.asarray(
            [self.desVel[4],
             self.desVel[5],
             self.desVel[6],
             0])

        trans_panel = tf.transformations.quaternion_matrix(
            [self.goalPose.orientation.x,
             self.goalPose.orientation.y,
             self.goalPose.orientation.z,
             self.goalPose.orientation.w])

        trans_panel[0, 3] = self.goalPose.position.x
        trans_panel[1, 3] = self.goalPose.position.y
        trans_panel[2, 3] = self.goalPose.position.z

        # inv_panel = np.zeros([4, 4])
        # inv_panel[3, 3] = 1.0
        # inv_panel[0:3, 0:3] = np.transpose(trans_panel[0:3, 0:3])
        # inv_panel[0:3, 3] = np.dot((-1*inv_panel[0:3, 0:3]),
        #                          trans_panel[0:3, 3])

        #vel_world = np.dot(inv_panel, vel_panel)
        vel_world = np.dot(trans_panel, vel_panel)
        #vel_world_ee = np.dot(inv_panel, vel_panel_ee)
        vel_world_ee = np.dot(trans_panel, vel_panel_ee)

        euler_test = tf.transformations.euler_from_quaternion(
            [self.goalPose.orientation.x,
             self.goalPose.orientation.y,
             self.goalPose.orientation.z,
             self.goalPose.orientation.w])

        # rospy.loginfo('Vel World ' + str(vel_world[0])
        #               + ', ' + str(vel_world[1])
        #               + ', ' + str(vel_world[2]))

        trans_auv = tf.transformations.quaternion_matrix(
            [self.robotPose.orientation.x,
             self.robotPose.orientation.y,
             self.robotPose.orientation.z,
             self.robotPose.orientation.w])

        trans_auv[0, 3] = self.robotPose.position.x
        trans_auv[1, 3] = self.robotPose.position.y
        trans_auv[2, 3] = self.robotPose.position.z

        inv_mat_auv = np.zeros([4, 4])
        inv_mat_auv[3, 3] = 1.0
        inv_mat_auv[0:3, 0:3] = np.transpose(trans_auv[0:3, 0:3])
        inv_mat_auv[0:3, 3] = np.dot((-1*inv_mat_auv[0:3, 0:3]),
                                     trans_auv[0:3, 3])

        vel_auv = np.dot(inv_mat_auv, vel_world)
        vel_arm = np.dot(inv_mat_auv, vel_world_ee)

        # rospy.loginfo('Vel auv ' + str(vel_auv[0])
        #               + ', ' + str(vel_auv[1])
        #               + ', ' + str(vel_auv[2]))

        vel_com = BodyVelocityReq()
        vel_com.header.stamp = rospy.get_rostime()
        vel_com.goal.priority = 10
        #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        vel_com.goal.requester = 'learning_algorithm'
        if not np.isnan(vel_auv[0]):
            if(abs(vel_auv[0]) <= 0.2):
                vel_com.twist.linear.x = vel_auv[0] #/50.0
            else:
                vel_com.twist.linear.x = np.sign(vel_auv[0])*0.2
        else:
            vel_com.twist.linear.x = 0.0

        if not np.isnan(vel_auv[1]):
            if(abs(vel_auv[1]) <= 0.2):
                vel_com.twist.linear.y = vel_auv[1] #/50.0
            else:
                vel_com.twist.linear.y = np.sign(vel_auv[1])*0.2
        else:
            vel_com.twist.linear.y = 0.0

        if not np.isnan(vel_auv[2]):
            if(abs(vel_auv[2]) <= 0.2):
                vel_com.twist.linear.z = vel_auv[2] #/30.0
            else:
                vel_com.twist.linear.z = np.sign(vel_auv[2])*0.2
        else:
            vel_com.twist.linear.z = 0.0

        if not np.isnan(self.desVel[3]):
            if(abs(self.desVel[3]) <= 0.2):
                vel_com.twist.angular.z = self.desVel[3]
            else:
                vel_com.twist.angular.z = np.sign(self.desVel[3])*0.2
        else:
            vel_com.twist.angular.z = 0.0

        #disabled_axis boby_velocity_req
        vel_com.disable_axis.x = False  # True False
        vel_com.disable_axis.y = False  # True False
        vel_com.disable_axis.z = False  # True False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = False # True False

        # rospy.loginfo('Desired Velocities X : '
        #               + str(vel_com.twist.linear.x)
        #               + ' Y: ' + str(vel_com.twist.linear.y)
        #               + ' Z: ' + str(vel_com.twist.linear.z)
        #               + ' Yaw: ' + str(vel_com.twist.angular.z))

        # Work around: This will be published later
        #self.pub_auv_vel.publish(vel_com)

        ##############################################
        # Compute the Arm velocity
        ##############################################
        joyCommand = Joy()
        # joyCommand.axes.append(vel_arm[0]*60.0)
        # joyCommand.axes.append(vel_arm[1]*60.0)
        # joyCommand.axes.append(vel_arm[2]*60.0)
        # rospy.loginfo('Vel Arm X ' + str(vel_arm[0]) + ' - ' + str(vel_auv[0]) + ' = ' + str(vel_arm[0]-vel_auv[0]))
        # rospy.loginfo('Vel Arm Y ' + str(vel_arm[1]) + ' - ' + str(vel_auv[1]) + ' = ' + str(vel_arm[1]-vel_auv[1]))
        # rospy.loginfo('Vel Arm Z ' + str(vel_arm[2]) + ' - ' + str(vel_auv[2]) + ' = ' + str(vel_arm[2]-vel_auv[2]))
        # rospy.loginfo('******************************************************')

        # COMMENT: Now this is done out-site the function
        # x_arm = (vel_arm[0]-vel_auv[0])
        # y_arm = (vel_arm[1]-vel_auv[1])
        # z_arm = (vel_arm[2]-vel_auv[2])

        # if np.abs(x_arm) > np.abs(y_arm) :
        #     if np.abs(x_arm) > np.abs(z_arm) :
        #         y_arm = y_arm/2.0
        #         z_arm = z_arm/2.0
        #     else :
        #         x_arm = z_arm/2.0
        #         y_arm = y_arm/2.0
        # else:
        #     if np.abs(y_arm) > np.abs(z_arm) :
        #         y_arm = y_arm/2.0
        #         z_arm = z_arm/2.0
        #     else :
        #         x_arm = x_arm/2.0
        #         y_arm = y_arm/2.0

        # rospy.loginfo('Original ' + str(vel_arm[0]) + ' ' +
        #               str(vel_arm[1]) + ' ' + str(vel_arm[2]))
        # x_arm = vel_arm[0] * 60
        # y_arm = vel_arm[1] * 60
        # z_arm = vel_arm[2] * 60
        x_arm = vel_arm[0]
        y_arm = vel_arm[1]
        z_arm = vel_arm[2]
        # rospy.loginfo('Vel Arm ' + str(x_arm) + ' ' +
        #               str(y_arm) + ' ' + str(z_arm))
        # rospy.loginfo('******************************************************')
        if not np.isnan(x_arm):
            joyCommand.axes.append(x_arm)
        else:
            rospy.loginfo('NAN NAN NAN NAN NAN')
            joyCommand.axes.append(0.0)
        if not np.isnan(y_arm):
            joyCommand.axes.append(y_arm)
        else:
            joyCommand.axes.append(0.0)
        if not np.isnan(z_arm):
            joyCommand.axes.append(z_arm)
        else:
            joyCommand.axes.append(0.0)
        if not np.isnan(self.desVel[7]):
            joyCommand.axes.append(self.desVel[7])
        else:
            joyCommand.axes.append(0.0)
        if not np.isnan(self.desVel[8]):
            joyCommand.axes.append(self.desVel[8])
        else:
            joyCommand.axes.append(0.0)
        if not np.isnan(self.desVel[9]):
            joyCommand.axes.append(-1.0*self.desVel[9])
            #joyCommand.axes.append(0.0)
        else:
            joyCommand.axes.append(0.0)



        #if self.limit_reach:
            #vel_com.twist.linear.x += joyCommand.axes[0]
            #vel_com.twist.linear.y += joyCommand.axes[1]
            #vel_com.twist.linear.z += joyCommand.axes[2]
            # force_movement_auv = 0.15
            # force_movement_ee = 0.15
            # rospy.loginfo('Vel AUV origin ' + str(vel_com.twist.linear.x) + ', ' + str(vel_com.twist.linear.y) + ', ' + str(vel_com.twist.linear.z) )
            # rospy.loginfo('Vel EE origin ' + str(joyCommand.axes[0]) + ', ' + str(joyCommand.axes[1]) + ', ' + str(joyCommand.axes[2]) )
            # if self.limit_reach_index < 0:
            #     #min
            #     vel_com.twist.linear.x -= force_movement_auv
            #     vel_com.twist.linear.y -= force_movement_auv
            #     joyCommand.axes[0] += force_movement_ee
            #     joyCommand.axes[2] -= (force_movement_ee + 0.05)
            #     rospy.logerr('AUV backward')
            # else: # self.limit_reach_index >0
            #     #max
            #     if self.limit_reach_index == 1:
            #         vel_com.twist.linear.x -= force_movement_auv
            #         joyCommand.axes[0] += force_movement_ee
            #         rospy.logerr('AUV backward')
            #     else: # index 1 i 2
            #         vel_com.twist.linear.z -= force_movement_auv
            #         joyCommand.axes[2] += force_movement_ee
            #         rospy.logerr('AUV Up')
            # rospy.loginfo('Vel AUV Modified ' + str(vel_com.twist.linear.x) + ', ' + str(vel_com.twist.linear.y) + ', ' + str(vel_com.twist.linear.z) )
            # rospy.loginfo('Vel EE Modified ' + str(joyCommand.axes[0]) + ', ' + str(joyCommand.axes[1]) + ', ' + str(joyCommand.axes[2]) )
            # rospy.loginfo('******************************************************')


        self.pub_auv_vel.publish(vel_com)
        self.pub_arm_command.publish(joyCommand)

        s = (repr(rospy.get_time()) + " " +
             repr(self.currPos[0]) + " " +
             repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " +
             repr(self.currPos[3]) + " " +
             repr(self.currPos[4]) + " " +
             repr(self.currPos[5]) + " " +
             repr(self.currPos[6]) + " " +
             repr(self.currPos[7]) + " " +
             repr(self.currPos[8]) + " " +
             repr(self.currPos[9]) + "\n")
             #repr(t) + "\n")
        self.file_export.write(s)

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_parametric_learning", "learning_reproductor_action.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_reproductor_action.yaml")

        rospy.init_node('learning_reproductor_action')
        learning_reproductor = learningReproductorAct(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
