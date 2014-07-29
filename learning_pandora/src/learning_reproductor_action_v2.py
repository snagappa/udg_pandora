#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('learning_pandora')
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
from sensor_msgs.msg import JointState

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
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy
from learning_pandora.msg import ValveTurningAction, ValveTurningFeedback
from learning_pandora.msg import ValveTurningResult

#import for the force torque sensor
from geometry_msgs.msg import WrenchStamped

from learning_pandora.msg import rfdm_msg
#from rfdm_pkg.msg import rfdm_msg

#from udg_pandora.srv import WorkAreaError

from cola2_control.srv import TurnDesiredDegrees, PushWithAUV

from cola2_control.srv import EFPose

import threading
import tf

from tf.transformations import euler_from_quaternion

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
        self.getLearnedParameters()
        self.goalPose = Pose()
        self.robotPose = Pose()
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
        self.dataGoalReceived = False
        self.dataGoalPoseReceived = False
        self.dataGoalOriReceived = False
        self.dataRobotReceived = False
        self.dataRollReceived = False
        self.dataComputed = 0
        #Simulation parameter
        self.currPosSim = np.zeros(self.nbVar)
        self.currPosSim[0] = 0.0
        self.currPosSim[1] = 0.0
        self.currPosSim[2] = 5.0
        self.currPosSim[3] = 0.0
        self.currPosSim[4] = 0.0
        self.currPosSim[5] = 0.0
        self.currPosSim[6] = 0.0

        self.unnormalized_angle = 0.0
        self.unnormalized_roll = 0.0

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

        if self.simulation:
            self.file = open(self.exportFile, 'w')
        else:
            self.fileTraj = open('real_traj.csv', 'w')

        self.fileAUVPose = open('auv_pose.csv', 'w')
        self.fileValvePose = open('panel_pose.csv', 'w')
        self.fileEFPose = open('ef_pose.csv', 'w')

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
        rospy.Subscriber(
            "/csip_e5_arm/joint_state",
            JointState,
            self.updateRollEndEffector,
            queue_size = 1)
        rospy.loginfo('Configuration ' + str(name) + ' Loaded ')

        #self.tflistener = tf.TransformListener()

        self.enable_srv = rospy.Service(
            '/learning/enable_reproductor_complete',
            Empty,
            self.enable_srv)

        self.disable_srv = rospy.Service(
            '/learning/disable_reproductor_complete',
            Empty,
            self.disable_srv)

        if self.force_torque_enable:
            rospy.loginfo('Force Torque Enabled ')
            rospy.Subscriber('/forceTorque_controller/forceTorqueData',
                             WrenchStamped,
                             self.updateForceTorque,
                             queue_size = 1)

        # self.valve_turning_srv = rospy.Service(
        #     '/learning/turn_valve_operation',
        #     TurnningValve,
        #     self.init_valve_turnning_srv)

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
        param_dict = {'reproductor_parameters': 'learning/reproductor/complete/parameters',
                      'alpha': 'learning/reproductor/complete/alpha',
                      's': 'learning/reproductor/complete/s',
                      'nbVar': 'learning/reproductor/complete/nbVar',
                      'interval_time': 'learning/reproductor/complete/interval_time',
                      'landmark_id': 'learning/reproductor/complete/landmark_id',
                      'interval_time': 'learning/reproductor/complete/interval_time',
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
                    # euler_test = tf.transformations.euler_from_quaternion(
                    #     [mark.pose.pose.orientation.x,
                    #      mark.pose.pose.orientation.y,
                    #      mark.pose.pose.orientation.z,
                    #      mark.pose.pose.orientation.w])
                    # rospy.loginfo('Euler test + ' + str(euler_test[0])
                    #               + ', ' + str(euler_test[1])
                    #               + ', ' + str(euler_test[2]))
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
            eul = euler_from_quaternion(
                [self.robotPose.orientation.x,
                 self.robotPose.orientation.y,
                 self.robotPose.orientation.z,
                 self.robotPose.orientation.w])
            s = (repr(self.robotPose.position.x) + " " +
                 repr(self.robotPose.position.y) + " " +
                 repr(self.robotPose.position.z) + " " +
                 repr(eul[0]) + " " +
                 repr(eul[1]) + " " +
                 repr(eul[2]) + "\n")
            self.fileAUVPose.write(s)
            if not self.dataRobotReceived:
                rospy.loginfo('Odometry Initialised')
                self.unnormalized_angle = euler_from_quaternion(
                    [self.robotPose.orientation.x,
                     self.robotPose.orientation.y,
                     self.robotPose.orientation.z,
                     self.robotPose.orientation.w])[2]
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
                # self.unnormalized_angle = self.unNormalizeAngle(
                #    self.unnormalized_angle, robotYaw)
                # self.currPos[3] = cola2_lib.normalizeAngle(
                #     goalYaw - robotYaw)
                # self.currPos[3] = goalYaw - self.unnormalized_angle

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

                self.currPos[3] = tf.transformations.euler_from_matrix(dif_ori)[2]

                # rospy.loginfo('Diff in roll ' + str(tf.transformations.euler_from_matrix(dif_ori)[0]))
                # rospy.loginfo('Diff in pitch ' + str(tf.transformations.euler_from_matrix(dif_ori)[1]))
                # rospy.loginfo('Diff in yaw ' + str(tf.transformations.euler_from_matrix(dif_ori)[2]))
                # rospy.loginfo('*************************************************')

                if self.dataReceived == 0:
                    self.currTime = (odometry.header.stamp.secs +
                                     (odometry.header.stamp.nsecs*1E-9))
                    self.dataReceived += 1

                elif self.dataReceived == 1:
                    #ERROR
                    self.prevPos[0:4] = self.currPos[0:4]
                    #####
                    self.prevTimeAUV = self.currTimeAUV
                    self.currTimeAUV = (odometry.header.stamp.secs +
                                        (odometry.header.stamp.nsecs*1E-9))
                    self.currVel[0:4] = ((self.currPos[0:4] - self.prevPos[0:4])
                                         / (self.currTimeAUV - self.prevTimeAUV))
                    self.dataReceived += 1
                else:
                    #ERROR
                    self.prevPos[0:4] = self.currPos[0:4]
                    #####
                    self.prevTimeAUV = self.currTimeAUV
                    self.currTimeAUV = (odometry.header.stamp.secs +
                                        (odometry.header.stamp.nsecs*1E-9))
                    self.currVel[0:4] = ((self.currPos[0:4] - self.prevPos[0:4]) /
                                         (self.currTimeAUV - self.prevTimeAUV))
            else:
                rospy.loginfo(
                    'Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()

    def updateRollEndEffector(self, joint_state):
        """
        This method is a work around to obatin only the orientation in the roll
        of the end effector. This way we simply the learning because the arm for
        the moment can't control only the Roll in the last joint.
        @param joint_state: Contains an array with the position of each joint.
        @type joint_state: JointState message from sensor_msgs
        """
        self.lock.acquire()
        try:
            if self.dataRollReceived :
                self.unnormalized_roll = self.unNormalizeAngle(
                    self.unnormalized_roll, joint_state.position[3])
            else :
                self.unnormalized_roll = joint_state.position[3]
                self.dataRollReceived = True
        finally:
            self.lock.release()

    def updateArmPosition(self, data):
        self.lock.acquire()
        try:
            if self.dataGoalReceived and self.dataRollReceived :
                endeffectorPose = np.array([data.pose.position.x,
                                            data.pose.position.y,
                                            data.pose.position.z,
                                            1])
                self.armOrientation = euler_from_quaternion([data.pose.orientation.x,
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
                s = (repr(self.armPose[0]) + " " +
                     repr(self.armPose[1]) + " " +
                     repr(self.armPose[2]) + " " +
                     repr(self.armOrientation[0]) + " " +
                     repr(self.armOrientation[1]) + " " +
                     repr(self.armOrientation[2]) + " " +
                     repr(rospy.get_time()) + "\n")
                self.fileEFPose.write(s)

                self.prevPos[4:10] = self.currPos[4:10]
                self.currPos[4:7] = self.armPose

                if self.valveOriInit:
                    self.currPos[9] = self.valveOri - self.unnormalized_roll
                else:
                    self.currPos[9] = self.unnormalized_roll
                self.currPos[7:9] = self.armOrientation[0:2]

                if self.dataReceivedArm == 0:
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.dataReceivedArm += 1
                elif self.dataReceivedArm == 1:
                    # ERROR
                    #self.prevPos[4:10] = self.currPos[4:10]
                    #######
                    self.prevTimeArm = self.currTimeArm
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.currVel[4:10] = ((self.currPos[4:10]-self.prevPos[4:10]) /
                                          (self.currTimeArm-self.prevTimeArm))
                    self.dataReceivedArm += 1
                else:
                    # ERROR
                    #self.prevPos[4:10] = self.currPos[4:10]
                    #######
                    self.prevTimeArm = self.currTimeArm
                    self.currTimeArm = (data.header.stamp.secs +
                                        (data.header.stamp.nsecs*1E-9))
                    self.currVel[4:10] = ((self.currPos[4:10]-self.prevPos[4:10]) /
                                          (self.currTimeArm-self.prevTimeArm))
            else:
                rospy.loginfo('Goal pose Not initialized')
        finally:
            self.lock.release()

    def updateSafety(self, rfdm_msg):
        self.lock.acquire()
        try:
            if (np.sign(self.action) == 1.0 or np.sign(self.action) == 0.0) and np.sign(rfdm_msg.reactive_data) == -1 :
                self.tf = -math.log(self.s)/self.alpha
                self.backward = True
            if np.sign(rfdm_msg.reactive_data) == 1.0 and (np.sign(self.action) == 1.0 or np.sign(self.action) == 0.0) :
                self.backward = False
                self.h_value = 0.0
            self.action = rfdm_msg.reactive_data
        finally:
            self.lock.release()

    def updateForceTorque(self, wrench_msg):
        self.lock_force.acquire()
        try:
            self.force_vector[0] = wrench_msg.wrench.force.x
            self.force_vector[1] = wrench_msg.wrench.force.y
            self.force_vector[2] = wrench_msg.wrench.force.z
            self.force_vector[3] = wrench_msg.wrench.torque.x
            self.force_vector[4] = wrench_msg.wrench.torque.y
            self.force_vector[5] = wrench_msg.wrench.torque.z
        finally:
            self.lock_force.release()


    def enable_srv(self, req):
        if not self.action_in_process :
            self.enabled = True
            rospy.loginfo('%s Enabled', self.name)
        else:
            rospy.loginfo('The %s is doing an action', self.name)
        return EmptyResponse()

    def disable_srv(self, req):
        if not self.action_in_process :
            self.enabled = False
            self.s = self.initial_s
            self.action = 1.0
            rospy.loginfo('%s Disabled', self.name)
        else:
            rospy.loginfo('The %s is doing an action', self.name)
        return EmptyResponse()

    def play(self):
#        pub = rospy.Publisher('arm', )
        rate = rospy.Rate(1.0/self.interval_time)
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
                            self.generateNewPose()
                    else:
                        self.simulatedNewPose()
                        # if self.currNbDataRepro >= self.nbDataRepro:
                        #     rospy.loginfo('Finish !!!!')
                        #     self.enabled = False
                        #     self.s = self.initial_s
                        #     rospy.signal_shutdown('The reproduction has finish')
            finally:
                self.lock.release()
            rate.sleep()
            # rospy.sleep(self.interval_time)

    def valve_turning_act(self, goal):
        """
        This function is call when a action is requested. The action function
        iterates until the DMP finish the execution or is preempted.
        @param goal: This param contains the id of the desired valve
        @type goal: ValveTurningAction
        """
        self.goal_valve = goal.valve_id
        self.sub_valve.unregister()
        self.sub_valve = rospy.Subscriber(('/valve_tracker/valve'+
                                           str(self.goal_valve)),
                                          PoseWithCovarianceStamped,
                                          self.updateGoalPose,
                                          queue_size = 1)

        #Set the id of the file which will be learned
        if goal.long_approach == True:
            self.learning_param_id = 0
        elif goal.long_approach == False:
            self.learning_param_id = 1

        rospy.loginfo('Start Action enable')

        #Load the learned data for the desired behaviour
        self.getLearnedParameters()
        #Restartin position data aboid multiple loop while waiting to call the action
        self.dataRollReceived = False
        self.dataReceivedArm = 0
        self.dataReceived = 0

        rate = rospy.Rate(1.0/self.interval_time)
        success = False
        preempted = False
        self.enabled = False
        self.action_in_process = True
        while not success and not preempted: #and self.force_big_update == 0:
            if self.dataReceived > 1 and self.dataReceivedArm > 1:
                if not self.simulation:
                    success = self.generateNewPose()
                    if self.force_torque_enable:
                        self.lock_force.acquire()
                        try:
                            #rospy.loginfo('Force in Z ' + str(self.force_vector[2]))
                            if np.abs(self.force_vector[2]) >= 20:
                                #self.force_big_update = 1
                                success = True
                        finally:
                            self.lock_force.release()
                else :
                    success = self.simulatedNewPose()
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
        # if preempted:
        #     result.valve_turned = False
        #     self.valve_turning_action.set_preempted()
        #     #retart the time
        #     self.s = self.initial_s
        # else :
        #     # Push until it touch the valve
        #     # Turn the valve and the desired degrees
        #     # stop pushing and go back
        #     # fold the arm
        #     # rospy.wait_for_service('/cola2_control/push_desired_froce')
        #     # rospy.wait_for_service('/cola2_control/turnDesiredRadians')
        #     # rospy.wait_for_service('/cola2_control/disable_push')
        #     try:
        #         push_srv = rospy.ServiceProxy('/cola2_control/push_desired_froce',
        #                                       PushWithAUV)
        #         rospy.loginfo('Pushing the valve ')
        #         push_srv = push_srv([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        #         # Keep stable force
        #         # if self.force_torque_enable:
        #         #     iterations = 0
        #         #     rate = rospy.Rate(1.0/self.interval_time)
        #         #     while iterations <= 20 and self.force_big_update < 5 and not rospy.is_shutdown():
        #         #         self.lock_force.acquire()
        #         #         try:
        #         #             if self.force_new_data:
        #         #                 if np.abs(self.force_vector_old[2] - self.force_vector[2]) > 20:
        #         #                     self.force_big_update = self.force_big_update + 1
        #         #                 else:
        #         #                     self.force_big_update = 0
        #         #                     self.force_vector_old = np.copy(self.force_vector)
        #         #                 self.force_new_data = False
        #         #         finally:
        #         #             self.lock_force.release()
        #         #         rate.sleep()
        #         # #wait until it feel the valve
        #         # else:
        #         #     rospy.sleep(4.0)

        #         rospy.sleep(4.0)

        #         turn_srv = rospy.ServiceProxy('/cola2_control/turnDesiredRadians',
        #                                        TurnDesiredDegrees)
        #         rospy.loginfo('desired increment ' + str(goal.desired_increment))
        #         res = turn_srv(goal.desired_increment)

        #         #stop the push
        #         dis_push_srv = rospy.ServiceProxy('/cola2_control/disable_push',
        #                                           Empty)

        #         dis_push_srv()
        #         rospy.loginfo('Stop Pushing and go backwards')
        #         #go backward
        #         rate = rospy.Rate(10)
        #         #rospy.loginfo('Init bucle')
        #         for i in range(80):
        #             #rospy.loginfo('Going backward')
        #             vel_com = BodyVelocityReq()
        #             vel_com.header.stamp = rospy.get_rostime()
        #             vel_com.goal.priority = 10
        #             #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        #             vel_com.goal.requester = 'learning_algorithm'
        #             vel_com.twist.linear.x = -0.1
        #             vel_com.twist.linear.y = 0.0
        #             vel_com.twist.linear.z = 0.0
        #             vel_com.twist.angular.z = 0.0

        #             #disabled_axis boby_velocity_req
        #             vel_com.disable_axis.x = False # True False
        #             vel_com.disable_axis.y = False # True False
        #             vel_com.disable_axis.z = True # True False
        #             vel_com.disable_axis.roll = True
        #             vel_com.disable_axis.pitch = True
        #             vel_com.disable_axis.yaw = False # True False
        #             self.pub_auv_vel.publish(vel_com)
        #             rate.sleep()

        #         # Fold the arm
        #         fold_arm_srv = rospy.ServiceProxy('/cola2_control/setPoseEF',
        #                                           EFPose)

        #         value = fold_arm_srv([0.45, 0.0, 0.11, 0.0, 0.0, 0.0 ])

        #         for i in range(80):
        #             #rospy.loginfo('Going backward')
        #             vel_com = BodyVelocityReq()
        #             vel_com.header.stamp = rospy.get_rostime()
        #             vel_com.goal.priority = 10
        #             #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        #             vel_com.goal.requester = 'learning_algorithm'
        #             vel_com.twist.linear.x = -0.05
        #             vel_com.twist.linear.y = 0.0
        #             vel_com.twist.linear.z = 0.0
        #             vel_com.twist.angular.z = 0.0

        #             #disabled_axis boby_velocity_req
        #             vel_com.disable_axis.x = False # True False
        #             vel_com.disable_axis.y = False # True False
        #             vel_com.disable_axis.z = True # True False
        #             vel_com.disable_axis.roll = True
        #             vel_com.disable_axis.pitch = True
        #             vel_com.disable_axis.yaw = False # True False
        #             self.pub_auv_vel.publish(vel_com)
        #             rate.sleep()

            #     rospy.loginfo('Finish')
            #     result.valve_turned = res.success

            # except rospy.ServiceException, e:
            #     print "Service call failed: %s"%e

        # JUST FOR TEST
        rospy.loginfo('Finish')
        result.valve_turned = res.success


        self.enabled = False
        self.s = self.initial_s
        self.action = 1.0

        #Stop all the movement sending Zero velocities
        joy_command = Joy()
        joy_command.axes.append(0.0)
        joy_command.axes.append(0.0)
        joy_command.axes.append(0.0)
        joy_command.axes.append(0.0)
        joy_command.axes.append(0.0)
        joy_command.axes.append(0.0)
        self.pub_arm_command.publish(joy_command)
        rospy.loginfo('Joy Message stop sent !!!!!')

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
        vel_com.disable_axis.x = True # True False
        vel_com.disable_axis.y = True # True False
        vel_com.disable_axis.z = True # True False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = True # True False
        self.pub_auv_vel.publish(vel_com)

        self.action_in_process = False
        self.valve_turning_action.set_succeeded(result)

    def simulatedNewPose(self):
        #rospy.loginfo('S : ' + str(self.s))
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = np.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        #rospy.loginfo('Vavlues on h ' + str(h))
        #rospy.loginfo('H Real ' + str(h.tolist()))
        if self.h_value > h[self.numStates-1]:
            rospy.loginfo('New end condition')
            self.enabled = False
            self.s = self.initial_s
            self.pub_auv_finish.publish(True)
            return True
        else:
            self.h_value = h[self.numStates-1]
            h = h / np.sum(h)

        # if np.sum(h) <= 0.0001:
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

        rospy.loginfo('H Norm ' + str(h.tolist()))

        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #rospy.loginfo( 'CurrWp \n' + currWp )
        #rospy.loginfo( 'CurrWp \n' + currWp )
        # rospy.loginfo('Kv ' + str(self.kV.tolist()))
        # rospy.loginfo('Curr Vel' + str(self.currVel.tolist()))
        # rospy.loginfo('Res ' + str((self.kV*self.currVel).tolist()))
        #rospy.loginfo('Curr Tar ' + str(currTar[0:3].tolist()))
        self.currAcc = ((np.dot(
            currWp, (currTar - self.currPosSim))) - (self.kV*self.currVel))

        #rospy.loginfo('Curr Diff ' + str((currTar - self.currPosSim)[0:3].tolist()))
        #rospy.loginfo('Curr Wp ' + str(currWp[0:3, 0:3]))
        #rospy.loginfo('Curr Dot ' + str((np.dot(
        #    currWp, (currTar - self.currPosSim)))[0:3].tolist()))
        #rospy.loginfo('Curr Acc ' + str(self.currAcc[0:3].tolist()))
        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)

        #rospy.loginfo('Pos ' +str(self.desPos[0]) + ' ' +str(self.desPos[1]) + ' ' +str(self.desPos[2]))
        des_pose_msg = PoseStamped()
        des_pose_msg.header.stamp = rospy.get_rostime()
        des_pose_msg.header.frame_id = "valve2"
        des_pose_msg.pose.position.x = self.desPos[0]
        des_pose_msg.pose.position.y = self.desPos[1]
        des_pose_msg.pose.position.z = self.desPos[2]
        des_pose_msg.pose.orientation.x = 0
        des_pose_msg.pose.orientation.y = 0
        des_pose_msg.pose.orientation.z = 0
        des_pose_msg.pose.orientation.w = 1

        self.pub_arm_des_pose.publish(des_pose_msg)

        s = (repr(self.desPos[0]) + " " +
             repr(self.desPos[1]) + " " +
             repr(self.desPos[2]) + " " +
             repr(self.desPos[3]) + " " +
             repr(self.desPos[4]) + " " +
             repr(self.desPos[5]) + " " +
             repr(self.desPos[6]) + " " +
             repr(self.desPos[7]) + " " +
             repr(self.desPos[8]) + " " +
             repr(self.desPos[9]) + " " +
             repr(rospy.get_time()) + " "+
             repr(t) + "\n")
        self.file.write(s)

        # why the interval time is here ????
        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time
        #rospy.loginfo('S - Salpah : ' + str(self.s) + '- ' + str(self.alpha*self.s*self.interval_time))
        self.currNbDataRepro = self.currNbDataRepro+1
        self.currPosSim = self.desPos

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
        if not self.backward and self.h_value > h[self.numStates-1]:
            rospy.loginfo('The time used in the demonstration is exhausted')
            self.enabled = False
            self.s = self.initial_s
            self.pub_auv_finish.publish(True)
            return True
        else:
            self.h_value = h[self.numStates-1]
            h = h / np.sum(h)

        rospy.loginfo('H values ' + str(h.tolist()))

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

        desPose_msg = PoseStamped()
        desPose_msg.header.stamp = rospy.get_rostime()
        desPose_msg.header.frame_id = "valve2"
        desPose_msg.pose.position.x = self.desPos[0]
        desPose_msg.pose.position.y = self.desPos[1]
        desPose_msg.pose.position.z = self.desPos[2]
        desPose_msg.pose.orientation.x = 0
        desPose_msg.pose.orientation.y = 0
        desPose_msg.pose.orientation.z = 0
        desPose_msg.pose.orientation.w = 1
        self.pub_arm_des_pose.publish(desPose_msg)

        self.publishCommands()

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
        # vel_auv2 = np.dot(inv_mat_auv, vel_world)
        vel_arm = np.dot(inv_mat_auv, vel_world_ee)

        # rospy.loginfo('Vel auv ' + str(vel_auv[0])
        #               + ', ' + str(vel_auv[1])
        #               + ', ' + str(vel_auv[2]))

        # if vel_auv[0] > 1.0:
        #     vel_auv[0] = 1.0
        # elif vel_auv[0] < -1.0:
        #     vel_auv[0] = -1.0

        vel_com = BodyVelocityReq()
        vel_com.header.stamp = rospy.get_rostime()
        vel_com.goal.priority = 10
        #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        vel_com.goal.requester = 'learning_algorithm'
        if not np.isnan(vel_auv[0]):
            if(abs(vel_auv[0]) <= 0.1):
                vel_com.twist.linear.x = vel_auv[0] #/50.0
            else:
                vel_com.twist.linear.x = np.sign(vel_auv[0])*0.1
        else:
            vel_com.twist.linear.x = 0.0

        if not np.isnan(vel_auv[1]):
            if(abs(vel_auv[1]) <= 0.1):
                vel_com.twist.linear.y = vel_auv[1] #/50.0
            else:
                vel_com.twist.linear.y = np.sign(vel_auv[1])*0.1
        else:
            vel_com.twist.linear.y = 0.0

        if not np.isnan(vel_auv[2]):
            if(abs(vel_auv[2]) <= 0.07):
                vel_com.twist.linear.z = vel_auv[2] #/30.0
            else:
                vel_com.twist.linear.z = np.sign(vel_auv[2])*0.07
        else:
            vel_com.twist.linear.z = 0.0

        if not np.isnan(self.desVel[3]):
            if(abs(vel_auv[2]) <= 0.05):
                vel_com.twist.angular.z = self.desVel[3]
            else:
                vel_com.twist.angular.z = np.sign(self.desVel[3])*0.05
        else:
            vel_com.twist.angular.z = 0.0

        #disabled_axis boby_velocity_req
        vel_com.disable_axis.x = False  # True False
        vel_com.disable_axis.y = False  # True False
        vel_com.disable_axis.z = False  # True False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = False # True False

        rospy.loginfo('Desired Velocities X : '
                      + str(vel_com.twist.linear.x)
                      + ' Y: ' + str(vel_com.twist.linear.y)
                      + ' Z: ' + str(vel_com.twist.linear.z)
                      + ' Yaw: ' + str(vel_com.twist.angular.z))
        self.pub_auv_vel.publish(vel_com)

        ##############################################
        # Compute the Arm velocity
        ##############################################
# Test COMMENT
        joyCommand = Joy()
        # joyCommand.axes.append(vel_arm[0]*60.0)
        # joyCommand.axes.append(vel_arm[1]*60.0)
        # joyCommand.axes.append(vel_arm[2]*60.0)
        # rospy.loginfo('Vel Arm X ' + str(vel_arm[0]) + ' - ' + str(vel_auv[0]) + ' = ' + str(vel_arm[0]-vel_auv[0]))
        # rospy.loginfo('Vel Arm Y ' + str(vel_arm[1]) + ' - ' + str(vel_auv[1]) + ' = ' + str(vel_arm[1]-vel_auv[1]))
        # rospy.loginfo('Vel Arm Z ' + str(vel_arm[2]) + ' - ' + str(vel_auv[2]) + ' = ' + str(vel_arm[2]-vel_auv[2]))
        # rospy.loginfo('******************************************************')
        x_arm = (vel_arm[0]-vel_auv[0])*120.0
        y_arm = (vel_arm[1]-vel_auv[1])*100.0
        z_arm = (vel_arm[2]-vel_auv[2])*100.0
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

        # x_arm = vel_arm[0] * 60
        # y_arm = vel_arm[1] * 60
        # z_arm = vel_arm[2] * 60

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
        else:
            joyCommand.axes.append(0.0)

        #self.pub_arm_command.publish(joyCommand)

        s = (repr(self.currPos[0]) + " " +
             repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " +
             repr(self.currPos[3]) + " " +
             repr(self.currPos[4]) + " " +
             repr(self.currPos[5]) + " " +
             repr(self.currPos[6]) + " " +
             repr(self.currPos[7]) + " " +
             repr(self.currPos[8]) + " " +
             repr(self.currPos[9]) + " " +
             repr(rospy.get_time()) + "\n")
             #repr(t) + "\n")

        self.fileTraj.write(s)

    def getLearnedParameters(self):
        """
        This method loads the data from a .txt file generated by the dmp
        learning. This file have to be in the learning_data folder in the
        udg_pandora package.
        """
        #find the subdirectory in the packge
        path = roslib.packages.get_pkg_subdir('learning_pandora','learning_data',False)
        # choose the file of the list in the learning directory
        param_id = self.learning_param_id
        # build the path and the file name
        file_path = path + '/' + self.reproductor_parameters[param_id]
        #rospy.loginfo('Name file ' + str(file_path))
        #read the file
        logfile = open(file_path, "r").readlines()

        logfile = [word.strip() for word in logfile]
        for i in xrange(len(logfile)):
            if logfile[i] == 'kV':
                i += 1
                # Individual KV
                aux = logfile[i].split(' ')
                self.kV = np.zeros(self.nbVar)
                for j in xrange(self.nbVar):
                    self.kV[j] = float(aux[j])
                # Colective KV
                # self.kV = float(logfile[i])
            elif logfile[i] == 'kP':
                i += 1
                # Individual KV
                aux = logfile[i].split(' ')
                self.kP = np.zeros(self.nbVar)
                for j in xrange(self.nbVar):
                    self.kP[j] = float(aux[j])
                # Colective KP
                # self.kP = float(logfile[i])
            elif logfile[i] == 'Mu_t':
                i += 1
                aux = logfile[i].split(' ')
                self.numStates = len(aux)
                self.Mu_t = np.zeros(self.numStates)
                for j in xrange(self.numStates):
                    self.Mu_t[j] = float(aux[j])
            elif logfile[i] == 'Sigma_t':
                i += 1
                self.Sigma_t = np.zeros(self.numStates)
                for j in xrange(self.numStates):
                    self.Sigma_t[j] = float(logfile[i])
                    i += 2
            elif logfile[i] == 'Mu_x':
                i += 1
                self.Mu_x = np.zeros(shape=(self.nbVar, self.numStates))
                for k in xrange(self.nbVar):
                    aux = logfile[i].split(' ')
                    for j in xrange(self.numStates):
                        self.Mu_x[k, j] = float(aux[j])
                    i += 1
            elif logfile[i] == 'Wp':
                i += 1
                self.Wp = np.zeros(
                    shape=(self.numStates, self.nbVar, self.nbVar))
                for z in xrange(self.numStates):
                    for k in xrange(self.nbVar):
                        aux = logfile[i].split(' ')
                        for j in xrange(self.nbVar):
                            self.Wp[z, k, j] = float(aux[j])
                        i += 1
                    i += 1
            else:
                pass

    def gaussPDF(self, Data, Mu, Sigma):
###     This function computes the Probability Density Function (PDF) of a
###     multivariate Gaussian represented by means and covariance matrix.
###
###     Author:	Sylvain Calinon, 2009
###             http://programming-by-demonstration.org
###
###     Inputs -----------------------------------------------------------------
###         o Data:  D x N array representing N datapoints of D dimensions.
###         o Mu:    D x K array representing the centers of the K GMM components.
###         o Sigma: D x D x K array representing the covariance matrices of the
###                  K GMM components.
###     Outputs ----------------------------------------------------------------
###         o prob:  1 x N array representing the probabilities for the
###                  N datapoints.
        if np.shape(Data) == ():
            nbVar = 1
            nbData = 1
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data - np.tile(Mu, (nbData, 1))
            prob = (Data*(1/Sigma)) * Data
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(
                        np.power((2*math.pi), nbVar) *
                        (abs(Sigma)+np.finfo(np.double).tiny)))
            return prob
        else:
            [nbVar, nbData] = np.shape(Data)
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data.T - np.tile(Mu.T, (nbData, 1))
            prob = np.sum(
                np.dot(Data, np.linalg.inv(Sigma))
                * Data, axis=1)
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(np.power((2*math.pi), nbVar)
                              * (abs(np.linalg.det(Sigma))
                                 + np.finfo(np.double).tiny)))
            return prob

    def unNormalizeAngle(self, current_angle, new_angle):
        """
        This function unNormalize the Angle obtaining a continuous values
        avoiding the discontinuity, jumps from 3.14 to -3.14
        @param currentAngle: contain the current angle not normalized
        @type currentAngle: double
        @param newAngle: contain the new angle normalized
        @type newAngle: double
        """
        if abs(current_angle) > np.pi:
            #We are over one lap over
            norm_curr = cola2_lib.normalizeAngle(current_angle)
            if abs(new_angle - norm_curr) > np.pi :
                if new_angle < 0.0:
                    inc0 = -1.0*(-np.pi - new_angle)
                    inc1 = -1.0*(np.pi - norm_curr)
                else:
                    inc0 = -1.0*(np.pi - new_angle)
                    inc1 = (-np.pi - norm_curr)
                return current_angle + inc0 + inc1
            else :
                return current_angle + (new_angle-norm_curr)
        else:
            if abs(new_angle - current_angle) > np.pi:
                if new_angle < 0.0:
                    inc0 = -1.0*(-np.pi - new_angle)
                    inc1 = -1.0*(np.pi - current_angle)
                else:
                    inc0 = -1.0*(np.pi - new_angle)
                    inc1 = (-np.pi - current_angle)
                return current_angle + inc0 + inc1
            else:
                return new_angle


if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "learning_pandora", "learning_reproductor_action_v2.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_reproductor_complete.yaml")

        rospy.init_node('learning_reproductor_action')
        learning_reproductor = learningReproductorAct(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
