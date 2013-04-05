#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

import math
import numpy as np

#use to load the configuration function
import cola2_ros_lib
# import the message to know the position
from geometry_msgs.msg import PoseStamped
# include the message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry
#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Point
#from geometry_msgs.msg import Quaternion
#include message of the point
from sensor_msgs.msg import Joy
#include message to move forward or backward the arm
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
import threading
import tf
#import warnings
#value to show all the numbers in a matrix
# np.set_printoptions(threshold=100000)


class learningReproductor:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = Point()
        self.goalOrientation = np.zeros(3)
        self.robotPose = Odometry()
        self.armPose = np.zeros(3)
        self.armOrientation = np.zeros(3)
        self.prevPos = np.zeros(self.nbVar)
        self.prevTime = 0.0
        self.currTime = 0.0
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
        self.currNbDataRepro = 0
        self.action = 1

        if self.simulation:
            self.file = open(self.exportFile, 'w')
        else:
            self.fileTraj = open('real_traj.csv', 'w')
            self.desTraj = open('des_traj.csv', 'w')

        #Debugging
        # self.filePub = open( 'pub_arm_pose.csv', 'w' )
        # self.fileDistance = open( 'distance.csv', 'w')
        # self.filePoseGoal = open( 'pose_goal.csv', 'w')
        # self.filePoseArmRobot = open( 'pose_ar.csv', 'w')
        # self.fileDesiredPose = open('desired_pose.csv', 'w')

        self.lock = threading.Lock()
        self.pub_desired_position = rospy.Publisher(
            "/arm/desired_position", PoseStamped)
        self.pub_arm_command = rospy.Publisher(
            "/cola2_control/joystick_arm_data", Joy)

        rospy.Subscriber('/arm/pose_stamped',
                         PoseStamped,
                         self.updateArmPosition)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry,
                         self.updateRobotPose)
        rospy.Subscriber('/work_area/evaluation',
                         Float64,
                         self.updateAction)
        rospy.loginfo('Configuration ' + str(name) + ' Loaded ')
        self.tflistener = tf.TransformListener()

    def getConfig(self):
        param_dict = {'reproductor_parameters':
                      'learning/reproductor/parameters',
                      'alpha': 'learning/reproductor/alpha',
                      's': 'learning/reproductor/s',
                      'nbVar': 'learning/reproductor/nbVar',
                      'interval_time': 'learning/reproductor/interval_time',
                      'landmark_id': 'learning/reproductor/landmark_id',
                      'interval_time': 'learning/reproductor/interval_time',
                      'simulation': 'learning/reproductor/simulation',
                      'nbDataRepro': 'learning/reproductor/nbDataRepro',
                      'exportFile': 'learning/reproductor/exportFile',
                      'frame_id_goal': 'learning/reproductor/frame_id_goal',
                      'poseGoal_x': 'learning/reproductor/poseGoal_x',
                      'poseGoal_y': 'learning/reproductor/poseGoal_y',
                      'poseGoal_z': 'learning/reproductor/poseGoal_z',
                      'name_pub_demonstrate':
                      'learning/reproductor/name_pub_demonstrate',
                      'name_pub_done': 'learning/reproductor/name_pub_done',
                      'quaternion_x': 'learning/reproductor/quaternion_x',
                      'quaternion_y': 'learning/reproductor/quaternion_y',
                      'quaternion_z': 'learning/reproductor/quaternion_z',
                      'quaternion_w': 'learning/reproductor/quaternion_w'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.position
                    try:
                        trans, rot = self.tflistener.lookupTransform(
                            "world",
                            "panel_centre",
                            self.tflistener.getLatestCommonTime(
                                "world", "panel_centre"))
                        rotation_matrix = tf.transformations.quaternion_matrix(
                            rot)

                        goalPose = np.asarray([self.poseGoal_x,
                                               self.poseGoal_y,
                                               self.poseGoal_z,
                                               1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose)

                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]
                        self.goalOrientation = euler_from_quaternion(rot)
                    except tf.Exception:
                        rotation_matrix = tf.transformations.quaternion_matrix(
                            [self.quaternion_x, self.quaternion_y,
                             self.quaternion_z, self.quaternion_w])
                        goalPose = np.asarray([self.poseGoal_x,
                                               self.poseGoal_y,
                                               self.poseGoal_z,
                                               1])
                        goalPose_rot = np.dot(rotation_matrix, goalPose)
                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]
                        self.goalOrientation = euler_from_quaternion(
                            [self.quaternion_x, self.quaternion_y,
                             self.quaternion_z, self.quaternion_w])
                    self.dataGoalReceived = True
        finally:
            self.lock.release()

    def updateRobotPose(self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
            if not self.dataRobotReceived:
                rospy.loginfo('Odometry Initialised')
                self.dataRobotReceived = True
        finally:
            self.lock.release()

    def updateAction(self, action):
        self.lock.acquire()
        try:
            self.action = action.data
        finally:
            self.lock.release()

    def play(self):
#        pub = rospy.Publisher('arm', )
        while not rospy.is_shutdown():
            self.lock.acquire()
            try:
                if not self.simulation:
                    if self.dataReceived > 1:
                        self.generateNewPose()
                else:
                    self.simulatedNewPose()
                    if self.currNbDataRepro >= self.nbDataRepro:
                        rospy.loginfo('Finish !!!!')
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
        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.currAcc = ((np.dot(currWp, (currTar-self.currPosSim)))
                        - (self.kV*self.currVel))
        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)
        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)
#        self.publishJoyMessage()
        #write in a file
        s = (repr(self.desPos[0]) + " " +
             repr(self.desPos[1]) + " " +
             repr(self.desPos[2]) + " " +
             repr(self.desPos[3]) + " " +
             repr(self.desPos[4]) + " " +
             repr(self.desPos[5]) + "\n")
        self.file.write(s)

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

        # print 'New S'
        # print self.s
        # raw_input()

        self.currNbDataRepro = self.currNbDataRepro+1
        self.currPosSim = self.desPos

    def generateNewPose(self):
        rospy.loginfo('Action value ' + str(self.action))
        if self.action == 1:
            t = -math.log(self.s)/self.alpha
            # for each atractor or state obtain the weigh
            #rospy.loginfo('Time :' + str(t) )
            h = np.zeros(self.numStates)
            for i in xrange(self.numStates):
                h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
            # normalize the value
            if np.sum(h) == 0:
                rospy.loginfo(
                    'The time used in the demonstration is exhausted')
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
            self.desAcc = ((np.dot(currWp, (currTar-self.currPos)))
                           - (self.kV*self.currVel))
            #action is a scalar value to evaluate the safety
            #currAcc = currAcc * math.fabs(self.action)

            self.desVel = self.currVel + self.desAcc * self.interval_time
            self.desPos = self.currPos + self.desVel * self.interval_time

            self.publishJoyMessage()

            #self.s = (self.s + (-self.alpha*self.s)
            #          * self.interval_time*self.action)
            self.s = self.s + (-self.alpha*self.s)*self.interval_time
        else:
            self.retractArm()
            self.s = 1

    def publishJoyMessage(self):
        joyCommand = Joy()
        # trans, rot = self.tflstener.lookupTransform("girona500", "world", rospy.Time())
        # rotation_matrix = tf.transformations.quaternion_matrix(rot)
        # desired_pose = np.asarray([self.desPos[0], self.desPos[1], self.desPos[2], 1])
        # desired_pose_tf = np.dot(rotation_matrix, desired_pose)[:3]
#        rospy.loginfo('Desired pose ' + str(self.desPos[0]) +', '+ str(self.desPos[1]) +', '+ str(self.desPos[2]) )
        newArmPose_x = self.goalPose.x + self.desPos[0]
        newArmPose_y = self.goalPose.y + self.desPos[1]
        newArmPose_z = self.goalPose.z + self.desPos[2]

        #World orientation
        command_x = newArmPose_x - self.armPose[0]
        command_y = newArmPose_y - self.armPose[1]
        command_z = newArmPose_z - self.armPose[2]
        command_roll = self.desPos[3] - self.currPos[3]
        command_pitch = self.desPos[4] - self.currPos[4]
        command_yaw = self.desPos[5] - self.currPos[5]

        #rospy.loginfo('Command ' + str(command_x) + ', ' +
        #              str(command_y) + ', ' + str(command_z))
        trans, rot = self.tflistener.lookupTransform(
            "girona500", "world",
            self.tflistener.getLatestCommonTime("girona500", "world"))
#        euler = tf.transformations.euler_from_quaternion(rot)
#        rospy.loginfo('Euler: ' + str(euler))
        rotation_matrix = tf.transformations.quaternion_matrix(rot)
        command_pose = np.asarray([command_x, command_y, command_z, 1])
        command_pose_tf = np.dot(rotation_matrix, command_pose)[:3]

        command_ori = np.asarray([command_roll, command_pitch, command_yaw, 1])
        command_ori_tf = np.dot(rotation_matrix, command_ori)[:3]

        # rospy.loginfo('Command Oriented ' + str(command_tf[0]) + ', ' +
        #               str(command_tf[1]) + ', ' + str(command_tf[2]))
        joyCommand.axes.append(command_pose_tf[0])
        joyCommand.axes.append(command_pose_tf[1])
        joyCommand.axes.append(command_pose_tf[2])
        joyCommand.axes.append(command_ori_tf[0])
        joyCommand.axes.append(command_ori_tf[1])
        joyCommand.axes.append(command_ori_tf[2])

        # Files to debug.
        #s = (repr(command_x) + " " + repr(command_y) +
        #     " " + repr(command_z) + "\n")
        # self.filePub.write(s)
        # s = (repr(self.goalPose.x -
        #           (self.armPose.pose.position.x +
        #            self.robotPose.pose.pose.position.x)) + " " +
        #      repr(self.goalPose.y -
        #           (self.armPose.pose.position.y +
        #            self.robotPose.pose.pose.position.y)) + " " +
        #      repr(self.goalPose.z -
        #           (self.armPose.pose.position.z +
        #            self.robotPose.pose.pose.position.z)) + "\n")
        # self.fileDistance.write(s)

        #s = (repr(self.goalPose.x) + " " + repr(self.goalPose.y) + " " +
        #     repr(self.goalPose.z) + "\n")
        # self.filePoseGoal.write(s)
        # s =(repr(self.armPose.pose.position.x + self.armPose.pose.position.x)
        #     + " " +
        #     repr(self.armPose.pose.position.y + self.armPose.pose.position.y)
        #     + " " +
        #     repr(self.armPose.pose.position.z + self.armPose.pose.position.z)
        #     + "\n"
        # self.filePoseArmRobot.write(s)
        # s = (repr(self.desPos[0]) + " " +
        #      repr(self.desPos[1]) + " " +
        #      repr(self.desPos[2]) + "\n")
        # self.fileDesiredPose.write(s)
        # Publish the map to create the path
        # pos_nav = PoseStamped()
        # pos_nav.header.frame_id = self.frame_id_goal
        # pos_nav.pose.position.x = float(self.desPos[0])
        # pos_nav.pose.position.y = float(self.desPos[1])
        # pos_nav.pose.position.z = float(self.desPos[2])
        #orientation, is needed a conversion from euler to quaternion
        # pos_nav.pose.point.position.x = pose_aux[0]
        # pos_nav.pose.point.position.y = pose_aux[1]
        # pos_nav.pose.point.position.z = pose_aux[2]
        #add the pose, point to the path
        # self.traj.poses.append(pos_nav)
        # self.pub_path_trajectory.publish(self.traj)

        s = (repr(self.currPos[0]) + " " + repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " + repr(self.currPos[3]) + " " +
             repr(self.currPos[4]) + " " + repr(self.currPos[5]) + "\n")
        self.fileTraj.write(s)

        s = (repr(newArmPose_x) + " " + repr(newArmPose_y) + " " +
             repr(newArmPose_z) + " " + repr(self.desPos[3]) + " " +
             repr(self.desPos[4]) + " " + repr(self.desPos[5]) + "\n")
        self.desTraj.write(s)
        self.pub_arm_command.publish(joyCommand)

    #Retract the Arm sending the command -x
    def retractArm(self):
        joyCommand = Joy()
        joyCommand.axes.append(-999)
        joyCommand.axes.append(-999)
        joyCommand.axes.append(-999)
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
        s = (repr(self.currPos[0]) + " " + repr(self.currPos[1]) +
             " " + repr(self.currPos[2]) + "\n")
        self.fileTraj.write(s)
        self.pub_arm_command.publish(joyCommand)

    def updateArmPosition(self, data):
        self.lock.acquire()
        try:
            # self.armPose = data
            # trans, rot = self.tflistener.lookupTransform(
            #     "world", "girona500", rospy.Time())
            # rotation_matrix = tf.transformations.quaternion_matrix(rot)
            # arm_pose = np.asarray([self.armPose.pose.position.x,
            #                        self.armPose.pose.position.y,
            #                        self.armPose.pose.position.z,
            #                        1])
            # arm_pose_tf = np.dot(rotation_matrix, arm_pose)[:3]
            arm_pose_tf, rot = self.tflistener.lookupTransform(
                "world", "end_effector",
                self.tflistener.getLatestCommonTime("world", "end_effector"))
            self.armPose = arm_pose_tf
            self.armOrientation = euler_from_quaternion(rot)
            #In the angles between the end EE and the valve are changed
            # Roll is the difference between Pitch in the world
            # Pitch is the difference between Roll in the world
            # Yaw is the differences between the Yaw in the world
            if self.dataRobotReceived and self.dataGoalReceived:
                if self.dataReceived == 0:
                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z
                    self.currPos[3] = (self.armOrientation[1] -
                                       self.goalOrientation[1])
                    self.currPos[4] = (self.armOrientation[0] -
                                       self.goalOrientation[0])
                    self.currPos[5] = (self.armOrientation[2] -
                                       self.goalOrientation[2])
                    self.currTime = (data.header.stamp.secs +
                                     (data.header.stamp.nsecs*1E-9))
                    self.dataReceived += 1
                elif self.dataReceived == 1:
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z
                    self.currPos[3] = (self.armOrientation[1] -
                                       self.goalOrientation[1])
                    self.currPos[4] = (self.armOrientation[0] -
                                       self.goalOrientation[0])
                    self.currPos[5] = (self.armOrientation[2] -
                                       self.goalOrientation[2])
                    self.currTime = (data.header.stamp.secs +
                                     (data.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos-self.prevPos) /
                                    (self.currTime-self.prevTime))
                    self.dataReceived += 1
                else:
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime
                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z
                    self.currPos[3] = (self.armOrientation[1] -
                                       self.goalOrientation[1])
                    self.currPos[4] = (self.armOrientation[0] -
                                       self.goalOrientation[0])
                    self.currPos[5] = (self.armOrientation[2] -
                                       self.goalOrientation[2])
                    self.currTime = (data.header.stamp.secs +
                                     (data.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos-self.prevPos) /
                                    (self.currTime-self.prevTime))
            else:
                if not self.dataRobotReceived:
                    rospy.loginfo('Waiting to initialise the  robot position')
                if not self.dataGoalReceived:
                    rospy.loginfo('Waiting to initialise the goal position')
        finally:
            self.lock.release()

    def getLearnedParameters(self):
        logfile = open(self.reproductor_parameters, "r").readlines()
        logfile = [word.strip() for word in logfile]
#        self.Mu_t = np.zeros( 3 )
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
#                rospy.loginfo('Values of the Mu_x \n' + str(self.Mu_x) )
            elif logfile[i] == 'Wp':
                i += 1
                self.Wp = np.zeros(shape=(self.numStates,
                                          self.nbVar,
                                          self.nbVar))
                for z in xrange(self.numStates):
                    for k in xrange(self.nbVar):
                        aux = logfile[i].split(' ')
                        for j in xrange(self.nbVar):
                            self.Wp[z, k, j] = float(aux[j])
                        i += 1
                    i += 1
#                rospy.loginfo('Values of Wp ' + str(self.Wp))
            else:
                pass
                #rospy.loginfo( logfile[i] +' ja no vull llegir mes')

    def gaussPDF(self, Data, Mu, Sigma):
###     This function computes the Probability Density Function (PDF) of a
###     multivariate Gaussian represented by means and covariance matrix.
###
###     Author:	Sylvain Calinon, 2009
###             http://programming-by-demonstration.org
###
###  Inputs -----------------------------------------------------------------
###      o Data:  D x N array representing N datapoints of D dimensions.
###      o Mu:    D x K array representing the centers of the K GMM components.
###      o Sigma: D x D x K array representing the covariance matrices of the
###                  K GMM components.
###  Outputs ----------------------------------------------------------------
###         o prob:  1 x N array representing the probabilities for the
###                  N datapoints.
        if np.shape(Data) == ():
            nbVar = 1
            nbData = 1
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data - np.tile(Mu, (nbData, 1))
            prob = (Data*(1/Sigma)) * Data
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(np.power((2*math.pi), nbVar) *
                              (abs(Sigma)+np.finfo(np.double).tiny)))
            return prob
        else:
            [nbVar, nbData] = np.shape(Data)
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data.T - np.tile(Mu.T, (nbData, 1))
            prob = np.sum(np.dot(Data, np.linalg.inv(Sigma)) * Data, axis=1)
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(np.power((2*math.pi), nbVar) *
                              (abs(np.linalg.det(Sigma)) +
                               np.finfo(np.double).tiny)))
            return prob
#        Data = Data.T - np.tile(Mu.T,(nbData,1))
        #prob = sum((Data*inv(Sigma)).*Data, 2);
#        prob = np.sum( np.dot(Data,np.linalg.inv(Sigma)) * Data, axis=1)
        #realmin = np.finfo(np.double).tiny
        # prob = (math.exp(-0.5*prob) /
        #         math.sqrt((2*math.pi) ^ nbVar*(abs(np.linalg.det(Sigma))
        #                                        + np.finfo(np.double).tiny)))

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_reproductor.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_reproductor.yaml")

        rospy.init_node('learning_reproductor')
        learning_reproductor = learningReproductor(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
