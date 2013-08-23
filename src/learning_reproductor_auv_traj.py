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
nnfrom nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark

#include message of the point
#from sensor_msgs.msg import Joy

#include the message to send velocities to the robot
from auv_msgs.msg import BodyVelocityReq

#include message to show the trajectories demonstrated
from nav_msgs.msg import Path

import math
import numpy
#from scipy import interpolate

#To enable or disable the service
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float64

import threading
import tf

from tf.transformations import euler_from_quaternion

#import warnings

#value to show all the numbers in a matrix
# numpy.set_printoptions(threshold=100000)


class learningReproductor:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = Pose()
        self.robotPose = Odometry()
        self.armPose = PoseStamped()
        self.prevPos = numpy.zeros(self.nbVar)
        self.prevTime = 0.0
        self.currTime = 0.0
        self.currPos = numpy.zeros(self.nbVar)
        self.currVel = numpy.zeros(self.nbVar)
        self.currAcc = numpy.zeros(self.nbVar)
        self.desAcc = numpy.zeros(self.nbVar)
        self.dataReceived = 0
        self.dataGoalReceived = False
        self.dataRobotReceived = False
        self.dataComputed = 0
        #Simulation parameter
        self.currPosSim = numpy.zeros(self.nbVar)
        self.currPosSim[0] = 0.5
        self.currPosSim[1] = 0.05
        self.currPosSim[2] = 0.8
        self.currPosSim[3] = 2.1
        self.currNbDataRepro = 0
        self.enabled = True
        self.initial_s = self.s
        if self.simulation:
            self.file = open(self.exportFile, 'w')
        else:
            self.fileTraj = open('real_traj.csv', 'w')

        #Debugging
        # self.filePub = open( 'pub_arm_pose.csv', 'w' )
        # self.fileDistance = open( 'distance.csv', 'w')
        # self.filePoseGoal = open( 'pose_goal.csv', 'w')
        # self.filePoseArmRobot = open( 'pose_ar.csv', 'w')
        # self.fileDesiredPose = open('desired_pose.csv', 'w')

        self.lock = threading.Lock()
        self.pub_desired_position = rospy.Publisher(
            "/arm/desired_position", PoseStamped)
        self.pub_auv_vel = rospy.Publisher(
            "/cola2_control/body_velocity_req", BodyVelocityReq)
        self.list_demos = []
        for i in range(len(self.demonstrations)):
            self.list_demos.append(
                rospy.Publisher(
                    self.name_pub_demonstrate + "_" +
                    str(self.demonstrations[i]), Path))

        self.pub_path_trajectory = rospy.Publisher(self.name_pub_done, Path)

        self.traj = Path()
        self.traj.header.frame_id = self.frame_id_goal

        rospy.Subscriber("/pose_ekf_slam/map",
                         Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry, self.updateRobotPose)

        rospy.loginfo('Configuration ' + str(name) + ' Loaded ')

        self.tflistener = tf.TransformListener()

        self.enable_srv = rospy.Service(
            '/learning/enable_reproductor_auv_traj',
            Empty,
            self.enableSrv)

        self.disable_srv = rospy.Service(
            '/learning/disable_reproductor_auv_traj',
            Empty,
            self.disableSrv)

        self.enable_with_s_srv = rospy.Service(
            '/learning/enable_reproductor_auv_traj_with_s',
            Float64,
            self.enableWithSSrv)

#        self.loadDemonstration()

    def getConfig(self):
        param_dict = {'reproductor_parameters': 'learning/reproductor/auv/parameters',
                      'alpha': 'learning/reproductor/auv/alpha',
                      's': 'learning/reproductor/auv/s',
                      'nbVar': 'learning/reproductor/auv/nbVar',
                      'interval_time': 'learning/reproductor/auv/interval_time',
                      'landmark_id': 'learning/reproductor/auv/landmark_id',
                      'interval_time': 'learning/reproductor/auv/interval_time',
                      'simulation': 'learning/reproductor/auv/simulation',
                      'nbDataRepro': 'learning/reproductor/auv/nbDataRepro',
                      'exportFile': 'learning/reproductor/auv/exportFile',
                      'demonstration_file': 'learning/reproductor/auv/demonstration_file',
                      'demonstrations': 'learning/reproductor/auv/demonstrations',
                      'frame_id_goal': 'learning/reproductor/auv/frame_id_goal',
                      'name_pub_demonstrate': 'learning/reproductor/auv/name_pub_demonstrate',
                      'name_pub_done': 'learning/reproductor/auv/name_pub_done',
                      'quaternion_x': 'learning/reproductor/auv/quaternion_x',
                      'quaternion_y': 'learning/reproductor/auv/quaternion_y',
                      'quaternion_z': 'learning/reproductor/auv/quaternion_z',
                      'quaternion_w': 'learning/reproductor/auv/quaternion_w'}
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Interval time value: ' + str(self.interval_time))

#WARNING THIS HAS NOT SENSE
# THE UPDATE WILL HAVE TO BE UPDATED BY THE DETECTION NOT FROM THE UPDATE
    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.goalPose = mark.pose.pose
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
            if self.dataGoalReceived:
                if self.dataReceived == 0:
                    self.currPos[0] = (odometry.pose.pose.position.x
                                       - self.goalPose.position.x)
                    self.currPos[1] = (odometry.pose.pose.position.y -
                                       self.goalPose.position.y)
                    self.currPos[2] = (odometry.pose.pose.position.z -
                                       self.goalPose.position.z)
                    #Yaw
                    goalYaw = euler_from_quaternion(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])[2]
                    robotYaw = euler_from_quaternion(
                        [odometry.pose.pose.orientation.x,
                         odometry.pose.pose.orientation.y,
                         odometry.pose.pose.orientation.z,
                         odometry.pose.pose.orientation.w])[2]
                    self.currPos[3] = cola2_lib.normalizeAngle(
                        robotYaw - goalYaw)
                    self.currTime = (odometry.header.stamp.secs +
                                     (odometry.header.stamp.nsecs*1E-9))
                    self.dataReceived += 1

                elif self.dataReceived == 1:
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = (odometry.pose.pose.position.x -
                                       self.goalPose.position.x)
                    self.currPos[1] = (odometry.pose.pose.position.y -
                                       self.goalPose.position.y)
                    self.currPos[2] = (odometry.pose.pose.position.z -
                                       self.goalPose.position.z)

                    #Yaw
                    goalYaw = euler_from_quaternion(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])[2]
                    robotYaw = euler_from_quaternion(
                        [odometry.pose.pose.orientation.x,
                         odometry.pose.pose.orientation.y,
                         odometry.pose.pose.orientation.z,
                         odometry.pose.pose.orientation.w])[2]
                    self.currPos[3] = cola2_lib.normalizeAngle(
                        robotYaw - goalYaw)
                    self.currTime = (odometry.header.stamp.secs +
                                     (odometry.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos - self.prevPos)
                                    / (self.currTime - self.prevTime))

                    self.dataReceived += 1

                else:
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = (odometry.pose.pose.position.x -
                                       self.goalPose.position.x)
                    self.currPos[1] = (odometry.pose.pose.position.y -
                                       self.goalPose.position.y)
                    self.currPos[2] = (odometry.pose.pose.position.z -
                                       self.goalPose.position.z)

                    #Yaw
                    goalYaw = euler_from_quaternion(
                        [self.goalPose.orientation.x,
                         self.goalPose.orientation.y,
                         self.goalPose.orientation.z,
                         self.goalPose.orientation.w])[2]
                    robotYaw = euler_from_quaternion(
                        [odometry.pose.pose.orientation.x,
                         odometry.pose.pose.orientation.y,
                         odometry.pose.pose.orientation.z,
                         odometry.pose.pose.orientation.w])[2]
                    self.currPos[3] = cola2_lib.normalizeAngle(
                        robotYaw - goalYaw)

                    self.currTime = (odometry.header.stamp.secs +
                                     (odometry.header.stamp.nsecs*1E-9))
                    self.currVel = ((self.currPos - self.prevPos) /
                                    (self.currTime - self.prevTime))
            else:
                rospy.loginfo(
                    'Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()

    def enableSrv(self, req):
        self.enabled = True
        rospy.loginfo('%s Enabled', self.name)
        return EmptyResponse()

    def enableWithSSrv(self, req):
        self.s = req
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
        h = numpy.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        h = h / numpy.sum(h)

        #init to vectors
        currTar = numpy.zeros(self.nbVar)
        currWp = numpy.zeros(shape=(self.nbVar, self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #rospy.loginfo( 'CurrWp \n' + currWp )
        #rospy.loginfo( 'CurrWp \n' + currWp )
        self.currAcc = ((numpy.dot(
            currWp, (currTar - self.currPosSim))) - (self.kV*self.currVel))

        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)

        s = (repr(self.desPos[0]) + " " +
             repr(self.desPos[1]) + " " +
             repr(self.desPos[2]) + " " +
             repr(self.desPos[3]) + "\n")
        self.file.write(s)

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

        self.currNbDataRepro = self.currNbDataRepro+1
        self.currPosSim = self.desPos

    def generateNewPose(self):
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = numpy.zeros(self.numStates)
        for i in xrange(self.numStates):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        if numpy.sum(h) == 0:
            rospy.loginfo('The time used in the demonstration is exhausted')
            rospy.signal_shutdown(
                'The time used in the demonstration is exhausted')
        else:
            h = h / numpy.sum(h)

        #init to vectors
        currTar = numpy.zeros(self.nbVar)
        currWp = numpy.zeros(shape=(self.nbVar, self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.desAcc = (numpy.dot(
            currWp, (currTar-self.currPos))) - (self.kV*self.currVel)
        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)

        self.desVel = self.currVel + self.desAcc * self.interval_time
        #NOT needed
        self.desPos = self.currPos + self.desVel * self.interval_time

        self.publishVelocityAUV()

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

    def publishVelocityAUV(self):
        trans, rot = self.tflistener.lookupTransform(
            "girona500", "world",
            self.tflistener.getLatestCommonTime(
                "girona500", "world"))
        rotation_matrix = tf.transformations.quaternion_matrix(rot)
        vel = numpy.asarray(
            [self.desVel[0],
             self.desVel[1],
             self.desVel[2],
             1])
        vel_tf = numpy.dot(rotation_matrix, vel)[:3]

        vel_com = BodyVelocityReq()
        vel_com.header.stamp = rospy.get_rostime()
        vel_com.goal.priority = 10
        #auv_msgs.GoalDescriptor.PRIORITY_NORMAL
        vel_com.goal.requester = 'learning_algorithm'
        vel_com.twist.linear.x = vel_tf[0] / 25.0
        vel_com.twist.linear.y = vel_tf[1] / 25.0
        vel_com.twist.linear.z = vel_tf[2] / 35.0
        vel_com.twist.angular.z = self.desVel[3] / 25.0

        #disabled_axis boby_velocity_req
        vel_com.disable_axis.x = False
        vel_com.disable_axis.y = False
        vel_com.disable_axis.z = False
        vel_com.disable_axis.roll = True
        vel_com.disable_axis.pitch = True
        vel_com.disable_axis.yaw = False
#        vel_com.disable_axis.yaw = True

        s = (repr(self.currPos[0]) + " " +
             repr(self.currPos[1]) + " " +
             repr(self.currPos[2]) + " " +
             repr(self.currPos[3]) + "\n")
        self.fileTraj.write(s)

        self.pub_auv_vel.publish(vel_com)

    def getLearnedParameters(self):
        logfile = open(self.reproductor_parameters, "r").readlines()
        logfile = [word.strip() for word in logfile]

#        self.Mu_t = numpy.zeros( 3 )
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
                self.Mu_t = numpy.zeros(self.numStates)
                for j in xrange(self.numStates):
                    self.Mu_t[j] = float(aux[j])
            elif logfile[i] == 'Sigma_t':
                i += 1
                self.Sigma_t = numpy.zeros(self.numStates)
                for j in xrange(self.numStates):
                    self.Sigma_t[j] = float(logfile[i])
                    i += 2
            elif logfile[i] == 'Mu_x':
                i += 1
                self.Mu_x = numpy.zeros(shape=(self.nbVar, self.numStates))
                for k in xrange(self.nbVar):
                    aux = logfile[i].split(' ')
                    for j in xrange(self.numStates):
                        self.Mu_x[k, j] = float(aux[j])
                    i += 1
#                rospy.loginfo('Values of the Mu_x \n' + str(self.Mu_x) )
            elif logfile[i] == 'Wp':
                i += 1
                self.Wp = numpy.zeros(
                    shape=(self.numStates, self.nbVar, self.nbVar))
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
###     Inputs -----------------------------------------------------------------
###         o Data:  D x N array representing N datapoints of D dimensions.
###         o Mu:    D x K array representing the centers of the K GMM components.
###         o Sigma: D x D x K array representing the covariance matrices of the
###                  K GMM components.
###     Outputs ----------------------------------------------------------------
###         o prob:  1 x N array representing the probabilities for the
###                  N datapoints.
        if numpy.shape(Data) == ():
            nbVar = 1
            nbData = 1
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data - numpy.tile(Mu, (nbData, 1))
            prob = (Data*(1/Sigma)) * Data
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(
                        numpy.power((2*math.pi), nbVar) *
                        (abs(Sigma)+numpy.finfo(numpy.double).tiny)))
            return prob
        else:
            [nbVar, nbData] = numpy.shape(Data)
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data.T - numpy.tile(Mu.T, (nbData, 1))
            prob = numpy.sum(
                numpy.dot(Data, numpy.linalg.inv(Sigma))
                * Data, axis=1)
            prob = (math.exp(-0.5*prob) /
                    math.sqrt(numpy.power((2*math.pi), nbVar)
                              * (abs(numpy.linalg.det(Sigma))
                                 + numpy.finfo(numpy.double).tiny)))
            return prob
#        Data = Data.T - numpy.tile(Mu.T,(nbData,1))
        #prob = sum((Data*inv(Sigma)).*Data, 2);
#        prob = numpy.sum( numpy.dot(Data,numpy.linalg.inv(Sigma)) * Data, axis=1)
        #realmin = numpy.finfo(numpy.double).tiny
#        prob = math.exp(-0.5*prob) / math.sqrt((2*math.pi)^nbVar * (abs(numpy.linalg.det(Sigma))+numpy.finfo(numpy.double).tiny))

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_pandora", "learning_reproductor_auv.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_record_auv.yaml")

        rospy.init_node('learning_reproductor_auv_traj')
        learning_reproductor = learningReproductor(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
