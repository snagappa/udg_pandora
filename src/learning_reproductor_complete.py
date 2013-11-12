#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#use to load the configuration function
import cola2_ros_lib

#use to normalize the angle
import cola2_lib

# import the service to call the service
# Warnning I don't know if is needed may be can be seen directly
#from cola2_control.srv import MoveArmTo
# import the message to know the position
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose

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
from sensor_msgs.msg import Joy

from udg_pandora.srv import WorkAreaError

import threading
import tf

from tf.transformations import euler_from_quaternion

#import warnings

#value to show all the numbers in a matrix
# numpy
# .set_printoptions(threshold=100000)


class learningReproductor:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = Pose()
        self.robotPose = Pose()
        self.armPose = PoseStamped()
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
        self.dataGoalReceived = False
        self.dataRobotReceived = False
        self.dataComputed = 0
        #Simulation parameter
        self.currPosSim = np.zeros(self.nbVar)
        self.currPosSim[0] = 0.5
        self.currPosSim[1] = 0.05
        self.currPosSim[2] = 0.8
        self.currPosSim[3] = 2.1
        self.currPosSim[4] = 0.2
        self.currPosSim[5] = 0.05
        self.currPosSim[6] = 0.02

        self.currNbDataRepro = 0
        self.enabled = False
        self.initial_s = self.s

        self.valveOri = 0.0
        self.valveOriInit = False

        if self.simulation:
            self.file = open(self.exportFile, 'w')
        else:
            self.fileTraj = open('real_traj.csv', 'w')

        self.fileAUVPose = open('auv_pose.csv', 'w')
        self.fileValvePose = open('panel_pose.csv', 'w')
        self.fileEFPose = open('ef_pose.csv', 'w')

        self.lock = threading.Lock()
        self.pub_auv_vel = rospy.Publisher(
            "/cola2_control/body_velocity_req", BodyVelocityReq)
        self.pub_arm_command = rospy.Publisher(
            "/cola2_control/joystick_arm_ef_vel", Joy)
        self.pub_auv_finish = rospy.Publisher(
            "learning/auv_finish", Bool)

        rospy.Subscriber("/pose_ekf_slam/map",
                         Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry, self.updateRobotPose)
        rospy.Subscriber('/arm/pose_stamped',
                         PoseStamped,
                         self.updateArmPosition)
        rospy.loginfo('Configuration ' + str(name) + ' Loaded ')

        self.tflistener = tf.TransformListener()

        self.enable_srv = rospy.Service(
            '/learning/enable_reproductor_complete',
            Empty,
            self.enableSrv)

        self.disable_srv = rospy.Service(
            '/learning/disable_reproductor_complete',
            Empty,
            self.disableSrv)

    def getConfig(self):
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
                      'demonstration_file': 'learning/reproductor/complete/demonstration_file',
                      'demonstrations': 'learning/reproductor/complete/demonstrations',
                      'frame_id_goal': 'learning/reproductor/complete/frame_id_goal',
                      'name_pub_demonstrate': 'learning/reproductor/complete/name_pub_demonstrate',
                      'name_pub_done': 'learning/reproductor/complete/name_pub_done',
                      'poseGoal_x': 'learning/reproductor/complete/poseGoal_x',
                      'poseGoal_y': 'learning/reproductor/complete/poseGoal_y',
                      'poseGoal_z': 'learning/reproductor/complete/poseGoal_z'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Interval time value: ' + str(self.interval_time))

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
                        self.dataGoalReceived = True
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
                        self.dataGoalReceived = True
                    try:
                        trans, rot = self.tflistener.lookupTransform(
                            "world", "valve2",
                            self.tflistener.getLatestCommonTime(
                                "world", "valve2"))
                        self.valveOri = tf.transformations.euler_from_quaternion(rot)[2]
                        #The Orientation of the valve is jumping
                        #rospy.loginfo('valve2 Orientation ' + str(self.valveOri))
                        #self.valveOriInit = True
                    except tf.Exception:
                        pass
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry.pose.pose
            eul = euler_from_quaternion(
                [odometry.pose.pose.orientation.x,
                 odometry.pose.pose.orientation.y,
                 odometry.pose.pose.orientation.z,
                 odometry.pose.pose.orientation.w])
            s = (repr(odometry.pose.pose.position.x) + " " +
                 repr(odometry.pose.pose.position.y) + " " +
                 repr(odometry.pose.pose.position.z) + " " +
                 repr(eul[0]) + " " +
                 repr(eul[1]) + " " +
                 repr(eul[2]) + "\n")
            self.fileAUVPose.write(s)
            if not self.dataRobotReceived:
                rospy.loginfo('Odometry Initialised')
                self.dataRobotReceived = True
            if self.dataGoalReceived:
                trans_mat = tf.transformations.quaternion_matrix(
                    [self.goalPose.orientation.x,
                     self.goalPose.orientation.y,
                     self.goalPose.orientation.z,
                     self.goalPose.orientation.w])

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
                self.currPos[3] = cola2_lib.normalizeAngle(
                    goalYaw - robotYaw)
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
                                         / (self.currTimeArm - self.prevTimeArm))
                    self.dataReceived += 1
                else:
                    self.prevPos[0:4] = self.currPos[0:4]
                    self.prevTimeAUV = self.currTimeAUV
                    self.currTimeAUV = (odometry.header.stamp.secs +
                                        (odometry.header.stamp.nsecs*1E-9))
                    self.currVel[0:4] = ((self.currPos[0:4] - self.prevPos[0:4]) /
                                         (self.currTimeArm - self.prevTimeArm))
            else:
                rospy.loginfo(
                    'Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()

    def updateArmPosition(self, data):
        self.lock.acquire()
        try:
            self.armPose = np.array([data.pose.position.x,
                                     data.pose.position.y,
                                     data.pose.position.z])
            self.armOrientation = euler_from_quaternion(
                [data.pose.orientation.x,
                 data.pose.orientation.y,
                 data.pose.orientation.z,
                 data.pose.orientation.w])
            s = (repr(self.armPose[0]) + " " +
                 repr(self.armPose[1]) + " " +
                 repr(self.armPose[2]) + " " +
                 repr(self.armOrientation[0]) + " " +
                 repr(self.armOrientation[1]) + " " +
                 repr(self.armOrientation[2]) + "\n")
            self.fileEFPose.write(s)

            self.prevPos[4:10] = self.currPos[4:10]
            self.currPos[4:7] = self.armPose
            if self.valveOriInit:
                self.currPos[7] =  cola2_lib.normalizeAngle(self.valveOri - self.armOrientation[2])
            else:
                self.currPos[7] = self.armOrientation[2]
                #rospy.loginfo('Roll ' + str(self.armOrientation[2]))
            self.currPos[8:10] = self.armOrientation[1:3]

            if self.dataReceived == 0:
                self.currTimeArm = (data.header.stamp.secs +
                                    (data.header.stamp.nsecs*1E-9))
                self.dataReceived += 1
            elif self.dataReceived == 1:
                self.prevTimeArm = self.currTimeArm
                self.currTimeArm = (data.header.stamp.secs +
                                    (data.header.stamp.nsecs*1E-9))
                self.currVel[4:10] = ((self.currPos[4:10]-self.prevPos[4:10]) /
                                      (self.currTimeArm-self.prevTimeArm))
                self.dataReceived += 1
            else:
                self.prevTimeArm = self.currTimeArm
                self.currTimeArm = (data.header.stamp.secs +
                                    (data.header.stamp.nsecs*1E-9))
                self.currVel[4:10] = ((self.currPos[4:10]-self.prevPos[4:10]) /
                                      (self.currTimeArm-self.prevTimeArm))
        finally:
            self.lock.release()

    def enableSrv(self, req):
        self.enabled = True
        rospy.loginfo('%s Enabled', self.name)
        return EmptyResponse()

    def disableSrv(self, req):
        self.enabled = False
        self.s = self.initial_s
        rospy.loginfo('%s Disabled', self.name)
        return EmptyResponse()

    def play(self):
#        pub = rospy.Publisher('arm', )
        while not rospy.is_shutdown():
            self.lock.acquire()
            try:
                if self.enabled:
                    if not self.simulation:
                        if self.dataReceived > 1:
                            self.generateNewPose()
                    else:
                        self.simulatedNewPose()
                        if self.currNbDataRepro >= self.nbDataRepro:
                            rospy.loginfo('Finish !!!!')
                            self.enabled = False
                            self.s = self.initial_s
                            rospy.signal_shutdown('The reproduction has finish')
            finally:
                self.lock.release()
            rospy.sleep(self.interval_time)

    def simulatedNewPose(self):
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = np.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        h = h / np.sum(h)

        #init to vectors
        currTar = np.zeros(self.nbVar)
        currWp = np.zeros(shape=(self.nbVar, self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #rospy.loginfo( 'CurrWp \n' + currWp )
        #rospy.loginfo( 'CurrWp \n' + currWp )
        self.currAcc = ((np.dot(
            currWp, (currTar - self.currPosSim))) - (self.kV*self.currVel))

        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)

        s = (repr(self.desPos[0]) + " " +
             repr(self.desPos[1]) + " " +
             repr(self.desPos[2]) + " " +
             repr(self.desPos[3]) + " " +
             repr(self.desPos[4]) + " " +
             repr(self.desPos[5]) + " " +
             repr(self.desPos[6]) + " " +
             repr(self.desPos[7]) + " " +
             repr(self.desPos[8]) + " " +
             repr(self.desPos[9]) + "\n")
        self.file.write(s)

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

        self.currNbDataRepro = self.currNbDataRepro+1
        self.currPosSim = self.desPos

    def generateNewPose(self):
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = np.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        if np.sum(h) == 0:
            rospy.loginfo('The time used in the demonstration is exhausted')
            rospy.signal_shutdown(
                'The time used in the demonstration is exhausted')
        else:
            h = h / np.sum(h)

        #init to vectors
        currTar = np.zeros(self.nbVar)
        currWp = np.zeros(shape=(self.nbVar, self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.desAcc = (np.dot(
            currWp, (currTar-self.currPos))) - (self.kV*self.currVel)
        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)

        self.desVel = self.currVel + self.desAcc * self.interval_time
        #NOT needed
        self.desPos = self.currPos + self.desVel * self.interval_time

        self.publishCommands()

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

        #rospy.loginfo('Value of S ' + str(self.s))
        if (self.s < 1E-200):
            rospy.loginfo('!!!!!!!! AUV trajectory Finish !!!!!!!!!')
            self.enabled = False
            self.pub_auv_finish.publish(True)
            self.s = self.initial_s

    def publishCommands(self):
        ##############################################
        # Compute the AUV velocity
        ##############################################
        vel_panel = np.asarray(
            [self.desVel[0],
             self.desVel[1],
             self.desVel[2],
             1])

        trans_panel = tf.transformations.quaternion_matrix(
            [self.goalPose.orientation.x,
             self.goalPose.orientation.y,
             self.goalPose.orientation.z,
             self.goalPose.orientation.w])

        trans_panel[0, 3] = self.goalPose.position.x
        trans_panel[1, 3] = self.goalPose.position.y
        trans_panel[2, 3] = self.goalPose.position.z

        vel_world = np.dot(trans_panel, vel_panel)

        trans_auv = tf.transformations.quaternion_matrix(
            [self.robotPose.orientation.x,
             self.robotPose.orientation.y,
             self.robotPose.orientation.z,
             self.robotPose.orientation.w])

        trans_auv[0, 3] = self.robotPose.position.x
        trans_auv[1, 3] = self.robotPose.position.y
        trans_auv[2, 3] = self.robotPose.position.z

        inv_mat = np.zeros([4, 4])
        inv_mat[3, 3] = 1.0
        inv_mat[0:3, 0:3] = np.transpose(trans_auv[0:3, 0:3])
        inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                 trans_auv[0:3, 3])

        vel_auv = np.dot(inv_mat, vel_world)

        vel_com = BodyVelocityReq()
        vel_com.header.stamp = rospy.get_rostime()
        vel_com.goal.priority = 10
        #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        vel_com.goal.requester = 'learning_algorithm'
        vel_com.twist.linear.x = vel_auv[0]/70.0
        vel_com.twist.linear.y = vel_auv[1]/70.0
        vel_com.twist.linear.z = vel_auv[2]/30.0
        vel_com.twist.angular.z = -self.desVel[3]/40.0

        #disabled_axis boby_velocity_req
        vel_com.disable_axis.x = False
        vel_com.disable_axis.y = False
        vel_com.disable_axis.z = False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = False
        #vel_com.disable_axis.yaw = True

        self.pub_auv_vel.publish(vel_com)
        ##############################################
        # Compute the Arm velocity
        ##############################################

        joyCommand = Joy()
        joyCommand.axes.append(self.desVel[4]* 6.)
        joyCommand.axes.append(self.desVel[5]* 6.)
        joyCommand.axes.append(self.desVel[6]* 6.)
        joyCommand.axes.append(-self.desVel[7]*0.0)#/20.)
        joyCommand.axes.append(self.desVel[8])
        joyCommand.axes.append(self.desVel[9])
        self.pub_arm_command.publish(joyCommand)

        # rospy.loginfo('Desired Roll ' + str(self.desPos[7]))
        # rospy.loginfo('Current Roll ' + str(self.currPos[7]))
        # rospy.loginfo('Vel Roll ' + str(-self.desVel[7]/10.0))
        # rospy.loginfo('**************************************')

        s = (repr(self.currPos[0]) + " " +
             repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " +
             repr(self.currPos[3]) + " " +
             repr(self.currPos[4]) + " " +
             repr(self.currPos[5]) + " " +
             repr(self.currPos[6]) + " " +
             repr(self.currPos[7]) + " " +
             repr(self.currPos[8]) + " " +
             repr(self.currPos[9]) + "\n")
        self.fileTraj.write(s)

    def getLearnedParameters(self):
        logfile = open(self.reproductor_parameters, "r").readlines()
        logfile = [word.strip() for word in logfile]
        for i in xrange(len(logfile)):
            if logfile[i] == 'kV':
                i += 1
                self.kV = float(logfile[i])
            elif logfile[i] == 'kP':
                i += 1
                self.kP = float(logfile[i])
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

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_reproductor_complete.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_reproductor_complete.yaml")

        rospy.init_node('learning_reproductor_complete')
        learning_reproductor = learningReproductor(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
