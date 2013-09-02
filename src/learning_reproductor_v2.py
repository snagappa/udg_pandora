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
from geometry_msgs.msg import Pose
#from geometry_msgs.msg import Quaternion
#include message of the point
from sensor_msgs.msg import Joy
#include message to move forward or backward the arm
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion
import threading
import tf
#import warnings
#value to show all the numbers in a matrix
# np.set_printoptions(threshold=100000)

from std_srvs.srv import Empty, EmptyResponse

#use to normalize the angle
import cola2_lib

#use to call the service to a disred position
from cola2_control.srv import EFPose


class learningReproductor:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = Pose()
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
        self.retracting = False
        self.enabled = False
        self.initial_s = self.s
        if self.simulation:
            self.file = open(self.exportFile, 'w')
        else:
            self.fileTraj = open('real_traj.csv', 'w')
            self.desTraj = open('des_traj.csv', 'w')

        self.fileValvePose = open('valve_pose.csv', 'w')
        self.fileEFPose = open('ef_pose.csv', 'w')

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
            "/cola2_control/joystick_arm_ef", Joy)
        self.pub_arm_finish = rospy.Publisher(
            "learning/arm_finish", Bool)

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
        self.enable_srv = rospy.Service(
            '/learning/enable_reproductor_arm',
            Empty,
            self.enableSrv)

        self.disable_srv = rospy.Service(
            '/learning/disable_reproductor_arm',
            Empty,
            self.disableSrv)

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
                      'safe_pose_ef': 'learning/reproductor/safe_pose_ef'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.pose.pose
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

                        self.goalPose.position.x = (self.goalPose.position.x +
                                                    goalPose_rot[0])
                        self.goalPose.position.y = (self.goalPose.position.y +
                                                    goalPose_rot[1])
                        self.goalPose.position.z = (self.goalPose.position.z +
                                                    goalPose_rot[2])
                        self.goalOrientation = euler_from_quaternion(rot)
                    except tf.Exception:
                        rotation_matrix = tf.transformations.quaternion_matrix(
                            [self.goalPose.orientation.x,
                             self.goalPose.orientation.y,
                             self.goalPose.orientation.z,
                             self.goalPose.orientation.w])
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

                    eul = euler_from_quaternion(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])
                    s = (repr(self.goalPose.position.x) + " " +
                         repr(self.goalPose.position.y) + " " +
                         repr(self.goalPose.position.z) + " " +
                         repr(eul[0]) + " " +
                         repr(eul[1]) + " " +
                         repr(eul[2]) + "\n")
                    self.fileValvePose.write(s)
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
        #rospy.loginfo('Action value ' + str(self.action))
        if self.action > 0.0:
            if self.retracting:
                self.retracting = False
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

            #rospy.loginfo(' Value s ' + str(self.s))
            if (self.s < 1E-120):
                rospy.loginfo('!!!!!!!! Arm trajectory Finish !!!!!!!!!')
                self.enabled = False
                self.pub_arm_finish.publish(True)
                self.s = self.initial_s

        elif self.action == 0.0:
            if self.retracting:
                self.retracting = False
            # stop the arm 
            #self.desPos = self.
            self.stopTheArm()
            pass
        else:
            if not self.retracting:
                self.retractArm()
                self.s = self.initial_s
            else:
                s = (repr(self.currPos[0]) + " " + repr(self.currPos[1]) +
                     " " + repr(self.currPos[2]) + "\n")
                self.fileTraj.write(s)

    def publishJoyMessage(self):
        joyCommand = Joy()
        # trans, rot = self.tflstener.lookupTransform("girona500", "world", rospy.Time())
        # rotation_matrix = tf.transformations.quaternion_matrix(rot)
        # desired_pose = np.asarray([self.desPos[0], self.desPos[1], self.desPos[2], 1])
        # desired_pose_tf = np.dot(rotation_matrix, desired_pose)[:3]
#        rospy.loginfo('Desired pose ' + str(self.desPos[0]) +', '+ str(self.desPos[1]) +', '+ str(self.desPos[2]) )
        # newArmPose_x = self.goalPose.position.x + self.desPos[0]
        # newArmPose_y = self.goalPose.position.y + self.desPos[1]
        # newArmPose_z = self.goalPose.position.z + self.desPos[2]

        # #World orientation
        # command_x = newArmPose_x - self.armPose[0]
        # command_y = newArmPose_y - self.armPose[1]
        # command_z = newArmPose_z - self.armPose[2]
        # command_roll = 0.0

        command_x = self.desPos[0] - self.currPos[0]
        command_y = self.desPos[1] - self.currPos[1]
        command_z = self.desPos[2] - self.currPos[2]
        command_roll = 0.0
        # command_roll = cola2_lib.normalizeAngle(
        #     self.desPos[3] - self.currPos[3])
        command_pitch = 0.0
        # command_pitch = cola2_lib.normalizeAngle(
        #     self.desPos[4] - self.currPos[4])
        command_yaw = 0.0
        # command_yaw = cola2_lib.normalizeAngle(
        #     self.desPos[5] - self.currPos[5])

        joyCommand.axes.append(command_x)
        joyCommand.axes.append(command_y)
        joyCommand.axes.append(command_z)
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
        #joyCommand.axes.append(command_roll/6.0)
        #joyCommand.axes.append(command_pitch/6.0)
        #joyCommand.axes.append(command_yaw/6.0)

        rospy.loginfo('Desired Pose ' + str(self.desPos[0])
                      + ' ' + str(self.desPos[1])
                      + ' ' + str(self.desPos[2]))
        rospy.loginfo('Current Pose ' + str(self.currPos[0])
                      + ' ' + str(self.currPos[1])
                      + ' ' + str(self.currPos[2]))
        rospy.loginfo('Command ' + str(command_x)
                      + ' ' + str(command_y)
                      + ' ' + str(command_z)
                      + ' ' + str(command_roll))

        s = (repr(self.currPos[0]) + " " + repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " + repr(self.currPos[3]) + " " +
             repr(self.currPos[4]) + " " + repr(self.currPos[5]) + "\n")
        self.fileTraj.write(s)

        s = (repr(self.desPos[0]) + " " + repr(self.desPos[1]) + " " +
             repr(self.desPos[2]) + " " + repr(self.desPos[3]) + " " +
             repr(self.desPos[4]) + " " + repr(self.desPos[5]) + "\n")
        self.desTraj.write(s)
        self.pub_arm_command.publish(joyCommand)

    #Retract the Arm sending the command -x
    def retractArm(self):
        rospy.wait_for_service('/cola2_control/setPoseEF')
        try:
            poseEF_srv = rospy.ServiceProxy('/cola2_control/setPoseEF', EFPose)
            success = poseEF_srv(self.safe_pose_ef)
            if success:
                self.retracting = True
            else:
                rospy.logerr('The safe position is not reachable by the arm')
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e

        s = (repr(self.currPos[0]) + " " + repr(self.currPos[1]) +
             " " + repr(self.currPos[2]) + "\n")
        self.fileTraj.write(s)

    #Stop the arm in the current postion
    def stopTheArm(self):
        joyCommand = Joy()
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
        joyCommand.axes.append(0.0)
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
            #In the angles between the end EE and the valve are changed
            # Roll is the difference between Pitch in the world
            # Pitch is the difference between Roll in the world
            # Yaw is the differences between the Yaw in the world

            trans_base, rot_base = self.tflistener.lookupTransform(
                "world", "base_arm",
                self.tflistener.getLatestCommonTime(
                    "world", "base_arm"))

            goalPose = np.array(
                [self.goalPose.position.x,
                 self.goalPose.position.y,
                 self.goalPose.position.z,
                 1])
            trans_matrix = tf.transformations.quaternion_matrix(
                rot_base)
            trans_matrix[0, 3] = trans_base[0]
            trans_matrix[1, 3] = trans_base[1]
            trans_matrix[2, 3] = trans_base[2]

            #invert Matrix
            inv_mat = np.zeros([4, 4])
            inv_mat[3, 3] = 1.0
            inv_mat[0:3, 0:3] = np.transpose(trans_matrix[0:3, 0:3])
            inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                     trans_matrix[0:3, 3])

            goalTrans = np.dot(inv_mat, goalPose)
            if self.dataRobotReceived and self.dataGoalReceived:
                if self.dataReceived == 0:
                    self.currPos[0] = goalTrans[0] - self.armPose[0]
                    self.currPos[1] = goalTrans[1] - self.armPose[1]
                    self.currPos[2] = goalTrans[2] - self.armPose[2]
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

                    self.currPos[0] = goalTrans[0] - self.armPose[0]
                    self.currPos[1] = goalTrans[1] - self.armPose[1]
                    self.currPos[2] = goalTrans[2] - self.armPose[2]
                    self.currPos[3] = cola2_lib.normalizeAngle(
                        self.armOrientation[1] - self.goalOrientation[1])
                    self.currPos[4] = cola2_lib.normalizeAngle(
                        cola2_lib.normalizeAngle(
                            self.armOrientation[0] -
                            self.goalOrientation[0]) -
                        (math.pi/2.0))
                    self.currPos[5] = cola2_lib.normalizeAngle(
                        (self.armOrientation[2] -
                         self.goalOrientation[2]))
                    self.currTime = (data.header.stamp.secs +
                                     (data.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos-self.prevPos) /
                                    (self.currTime-self.prevTime))
                    self.dataReceived += 1
                else:
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime
                    self.currPos[0] = goalTrans[0] - self.armPose[0]
                    self.currPos[1] = goalTrans[1] - self.armPose[1]
                    self.currPos[2] = goalTrans[2] - self.armPose[2]
                    self.currPos[3] = cola2_lib.normalizeAngle(
                        self.armOrientation[1] - self.goalOrientation[1])
                    self.currPos[4] = cola2_lib.normalizeAngle(
                        cola2_lib.normalizeAngle(
                            self.armOrientation[0] -
                            self.goalOrientation[0]) -
                        (math.pi/2.0))
                    self.currPos[5] = cola2_lib.normalizeAngle(
                        (self.armOrientation[2] -
                         self.goalOrientation[2]))
                    self.currTime = (data.header.stamp.secs +
                                     (data.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos-self.prevPos) /
                                    (self.currTime-self.prevTime))

                rospy.loginfo('Diff ' + str(self.currPos[0])
                              + ' ' + str(self.currPos[1])
                              + ' ' + str(self.currPos[2]))
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
            "udg_pandora", "learning_reproductor_arm.yaml")
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
