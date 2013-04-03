#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#use to load the configuration function
import cola2_ros_lib
# import the service to call the service
# Warnning I don't know if is needed may be can be seen directly
from cola2_control.srv import MoveArmTo
# import the message to know the position
from geometry_msgs.msg import PoseStamped

# include the message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Point

#include message of the point
from sensor_msgs.msg import Joy

#include message to show the trajectories demonstrated
from nav_msgs.msg import Path

import math
import numpy
from scipy import interpolate

import threading
import tf

#import warnings

#value to show all the numbers in a matrix
# numpy.set_printoptions(threshold=100000)

class learningReproductor :

    def __init__(self , name) :
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = Point()
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
        self.currPosSim = numpy.zeros(3)
        self.currPosSim[0] = 0.5
        self.currPosSim[1] = 0.05
        self.currPosSim[2] = 0.8
        self.currNbDataRepro = 0

        if self.simulation : self.file = open( self.exportFile, 'w')
        else : self.fileTraj = open( 'real_traj.csv', 'w')

        #Debugging
        # self.filePub = open( 'pub_arm_pose.csv', 'w' )
        # self.fileDistance = open( 'distance.csv', 'w')
        # self.filePoseGoal = open( 'pose_goal.csv', 'w')
        # self.filePoseArmRobot = open( 'pose_ar.csv', 'w')
        # self.fileDesiredPose = open('desired_pose.csv', 'w')

        self.lock = threading.Lock()
        self.pub_desired_position = rospy.Publisher("/arm/desired_position", PoseStamped )
        self.pub_arm_command = rospy.Publisher("/cola2_control/joystick_arm_data", Joy )
        self.list_demos = []
        for i in  range(len(self.demonstrations)) :
            self.list_demos.append( rospy.Publisher( self.name_pub_demonstrate + "_" + str(self.demonstrations[i]), Path)  )

        self.pub_path_trajectory = rospy.Publisher( self.name_pub_done, Path)

        self.traj = Path()
        self.traj.header.frame_id = self.frame_id_goal


        rospy.Subscriber('/arm/pose_stamped', PoseStamped , self.updateArmPosition )
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )

        rospy.loginfo('Configuration ' + str(name) +  ' Loaded ')

        self.tflistener = tf.TransformListener()

#        self.loadDemonstration()

    def getConfig(self) :
        param_dict = {'reproductor_parameters': 'learning/reproductor/pose/parameters',
                      'alpha': 'learning/reproductor/pose/alpha',
                      's': 'learning/reproductor/pose/s',
                      'nbVar': 'learning/reproductor/pose/nbVar',
                      'interval_time': 'learning/reproductor/pose/interval_time',
                      'landmark_id': 'learning/reproductor/pose/landmark_id',
                      'interval_time': 'learning/reproductor/pose/interval_time',
                      'simulation': 'learning/reproductor/pose/simulation',
                      'nbDataRepro': 'learning/reproductor/pose/nbDataRepro',
                      'exportFile': 'learning/reproductor/pose/exportFile',
                      'demonstration_file': 'learning/reproductor/pose/demonstration_file',
                      'demonstrations': 'learning/reproductor/pose/demonstrations',
                      'frame_id_goal': 'learning/reproductor/pose/frame_id_goal',
                      'poseGoal_x': 'learning/record/pose/poseGoal_x',
                      'poseGoal_y': 'learning/record/pose/poseGoal_y',
                      'poseGoal_z': 'learning/record/pose/poseGoal_z',
                      'name_pub_demonstrate': 'learning/reproductor/pose/name_pub_demonstrate',
                      'name_pub_done': 'learning/reproductor/pose/name_pub_done'}
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Interval time value: ' + str(self.interval_time) )

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:
            for mark in landMarkMap.landmark :
                if self.landmark_id == mark.landmark_id :
                    #rospy.loginfo('Ha arribat alguna cosa')
                    try:
                        trans, rot = self.tflistener.lookupTransform("world", "panel_centre", self.tflistener.getLatestCommonTime( "world", "panel_centre" ))
                        rotation_matrix = tf.transformations.quaternion_matrix(rot)
                        goalPose = numpy.asarray([self.poseGoal_x, self.poseGoal_y, self.poseGoal_z, 1])
                        goalPose_rot = numpy.dot(rotation_matrix, goalPose)[:3]

                        self.goalPose = mark.position
                        self.goalPose.x = mark.position.x + goalPose_rot[0]
                        self.goalPose.y = mark.position.y + goalPose_rot[1]
                        self.goalPose.z = mark.position.z + goalPose_rot[2]

                        if not self.dataGoalReceived :
                            rospy.loginfo('Goal Pose Received')
                            self.dataGoalReceived = True

                    except tf.Expetion:
                        pass
        finally:
            self.lock.release()

    def updateRobotPose (self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
            if not self.dataRobotReceived :
                rospy.loginfo('Odometry Initialised')
                self.dataRobotReceived = True
        finally:
            self.lock.release()



    def play(self) :
#        pub = rospy.Publisher('arm', )
        while not rospy.is_shutdown():
            self.lock.acquire()
            try:
                if not self.simulation :
                    if  self.dataReceived >1 :
                        self.generateNewPose()
                else :
                    self.simulatedNewPose()
                    if self.currNbDataRepro >= self.nbDataRepro :
                        rospy.loginfo('Finish !!!!')
                        rospy.signal_shutdown('The reproduction has finish')
            finally:
                self.lock.release()
            rospy.sleep(self.interval_time)


    def simulatedNewPose(self) :
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = numpy.zeros(self.numStates)
        for i in xrange(self.numStates) :
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        h = h / numpy.sum(h)

        #init to vectors
        currTar = numpy.zeros(self.nbVar)
        currWp = numpy.zeros(shape=(self.nbVar,self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates ) :
            currTar = currTar + self.Mu_x[:,i]*h[i]
            currWp = currWp + self.Wp[i,:,:]*h[i]

        # print 'CurrTar'
        # print currTar

        # print 'CurrWp'
        # print currWp

        # print 'currVel'
        # print self.currVel

        # print 'CurrPosSim'
        # print self.currPosSim

        # print 'Kv'
        # print self.kV

        # raw_input()

        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.currAcc = (numpy.dot(currWp, (currTar-self.currPosSim))) - (self.kV*self.currVel)

        # print 'CurrAcc'
        # print self.currAcc

        # raw_input()


        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)

        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)

        # print 'Interval_time'
        # print self.interval_time

        # print 'CurrVel'
        # print self.currVel

        # print 'desPos'
        # print self.desPos

        # raw_input()

#        self.publishJoyMessage()
        #write in a file

        s = repr( self.desPos[0] ) + " " + repr( self.desPos[1]) +  " " + repr(self.desPos[2]) + "\n"
        self.file.write(s)

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

        # print 'New S'
        # print self.s
        # raw_input()

        self.currNbDataRepro = self.currNbDataRepro+1
        self.currPosSim = self.desPos



    def generateNewPose(self) :
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = numpy.zeros(self.numStates)
        for i in xrange(self.numStates) :
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        if numpy.sum(h) == 0 :
            rospy.loginfo('The time used in the demonstration is exhausted')
            rospy.signal_shutdown('The time used in the demonstration is exhausted')
        else :
            h = h / numpy.sum(h)

        #init to vectors
        currTar = numpy.zeros(self.nbVar)
        currWp = numpy.zeros(shape=(self.nbVar,self.nbVar))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates ) :
            currTar = currTar + self.Mu_x[:,i]*h[i]
            currWp = currWp + self.Wp[i,:,:]*h[i]

        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.desAcc = (numpy.dot(currWp, (currTar-self.currPos))) - (self.kV*self.currVel)
        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)

        self.desVel = self.currVel + self.desAcc * self.interval_time
        self.desPos = self.currPos + self.desVel * self.interval_time

        self.publishJoyMessage()
#        if self.dataComputed == 9 :
#            self.publishJoyMessage()
#            self.dataComputed = 0
#        else :
#            self.dataComputed += 1

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time


    def publishJoyMessage(self) :
        joyCommand = Joy()

        # trans, rot = self.tflstener.lookupTransform("girona500", "world", rospy.Time())
        # rotation_matrix = tf.transformations.quaternion_matrix(rot)
        # desired_pose = numpy.asarray([self.desPos[0], self.desPos[1], self.desPos[2], 1])
        # desired_pose_tf = numpy.dot(rotation_matrix, desired_pose)[:3]

#        rospy.loginfo('Desired pose ' + str(self.desPos[0]) +', '+ str(self.desPos[1]) +', '+ str(self.desPos[2]) )

        newArmPose_x = self.goalPose.x + self.desPos[0] # desired_pose_tf[0]
        newArmPose_y = self.goalPose.y + self.desPos[1] # desired_pose_tf[1]
        newArmPose_z = self.goalPose.z + self.desPos[2] # desired_pose_tf[2]

        #Debbuging the orientation
        # trans, rot = self.tflistener.lookupTransform("world", "girona500", rospy.Time())
        # rotation_matrix = tf.transformations.quaternion_matrix(rot)
        # rospy.loginfo('Rotation Matrix \n'+ str(rotation_matrix))
        # rospy.loginfo('Rotation of [1,0,0,1]: '+ str(numpy.dot(rotation_matrix,numpy.asarray([1,0,0,1])) ) )
        # rospy.loginfo('*******************************************************')


        rospy.loginfo('Desired Pose Converted  ' + str(newArmPose_x) +', '+ str(newArmPose_y) +', '+ str(newArmPose_z) )

        # trans, rot = self.tflistener.lookupTransform("world", "end_effector", rospy.Time())
        # rotation_matrix = tf.transformations.quaternion_matrix(rot)
        # arm_pose = numpy.asarray([self.armPose.pose.position.x, self.armPose.pose.position.y, self.armPose.pose.position.z, 1])
        # arm_pose_tf = numpy.dot(rotation_matrix, arm_pose)[:3]

        # currArmPose_x = arm_pose_tf[0] + self.robotPose.pose.pose.position.x
        # currArmPose_y = arm_pose_tf[1] + self.robotPose.pose.pose.position.y
        # currArmPose_z = arm_pose_tf[2] + self.robotPose.pose.pose.position.z

 #       rospy.loginfo('Valve Center pose  ' + str(self.goalPose.x) +', '+ str(self.goalPose.y) +', '+ str(self.goalPose.z) )
        #rospy.loginfo('Current pose ' + str(currArmPose_x) +', '+ str(currArmPose_y) +', '+ str(currArmPose_z) )

        rospy.loginfo('Current pose ' + str(self.armPose[0]) +', '+ str(self.armPose[1]) +', '+ str(self.armPose[2]) )

        #World orientation
        command_x = newArmPose_x - self.armPose[0]
        command_y = newArmPose_y - self.armPose[1]
        command_z = newArmPose_z - self.armPose[2]

        rospy.loginfo('Command ' + str(command_x) +', '+ str(command_y) +', '+ str(command_z) )

        trans, rot = self.tflistener.lookupTransform("girona500", "world", self.tflistener.getLatestCommonTime("girona500","world"))
#        euler = tf.transformations.euler_from_quaternion(rot)
#        rospy.loginfo('Euler: ' + str(euler))
        rotation_matrix = tf.transformations.quaternion_matrix(rot)
        command = numpy.asarray([command_x, command_y, command_z, 1])
        command_tf = numpy.dot(rotation_matrix, command)[:3]

        test = numpy.asarray([1, 0, 0, 1])
        rospy.loginfo('Translation ' + str(numpy.dot(rotation_matrix,test)))

        rospy.loginfo('Command Oriented ' + str(command_tf[0]) +', '+ str(command_tf[1]) +', '+ str(command_tf[2]) )
        rospy.loginfo('*******************************************************')

        joyCommand.axes.append( command_tf[0] )
        joyCommand.axes.append( command_tf[1] )
        joyCommand.axes.append( command_tf[2] )
        joyCommand.axes.append( 0.0 )
        joyCommand.axes.append( 0.0 )
        joyCommand.axes.append( 0.0 )


        # Files to debug.
        # s = repr( command_x ) + " " + repr( command_y ) +  " " + repr( command_z ) + "\n"
        # self.filePub.write(s)
        # s = repr(  self.goalPose.x - (self.armPose.pose.position.x + self.robotPose.pose.pose.position.x ) ) + " " + repr( self.goalPose.y - (self.armPose.pose.position.y + self.robotPose.pose.pose.position.y  )  ) +  " " + repr( self.goalPose.z - (self.armPose.pose.position.z + self.robotPose.pose.pose.position.z  ) ) + "\n"
        # self.fileDistance.write(s)
        # s = repr( self.goalPose.x) + " " + repr( self.goalPose.y ) +  " " + repr( self.goalPose.z ) + "\n"
        # self.filePoseGoal.write(s)
        # s = repr( self.armPose.pose.position.x + self.armPose.pose.position.x  ) + " " + repr( self.armPose.pose.position.y + self.armPose.pose.position.y  ) +  " " + repr( self.armPose.pose.position.z + self.armPose.pose.position.z  ) + "\n"
        # self.filePoseArmRobot.write(s)
        # s = repr( self.desPos[0] ) + " " + repr( self.desPos[1] ) +  " " + repr( self.desPos[2] ) + "\n"
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

        s = repr( self.currPos[0] ) + " " + repr( self.currPos[1]) +  " " + repr(self.currPos[2]) + "\n"
        self.fileTraj.write(s)

        self.pub_arm_command.publish(joyCommand)


    def updateArmPosition(self, data):
        self.lock.acquire()
        try:
            # self.armPose = data
            # trans, rot = self.tflistener.lookupTransform("world", "girona500", rospy.Time())
            # rotation_matrix = tf.transformations.quaternion_matrix(rot)
            # arm_pose = numpy.asarray([self.armPose.pose.position.x, self.armPose.pose.position.y, self.armPose.pose.position.z, 1])
            # arm_pose_tf = numpy.dot(rotation_matrix, arm_pose)[:3]

            arm_pose_tf, rot = self.tflistener.lookupTransform("world", "end_effector", self.tflistener.getLatestCommonTime("world","end_effector") )
            self.armPose = arm_pose_tf

            if self.dataRobotReceived and self.dataGoalReceived :
                if self.dataReceived == 0 :

                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z

                    self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                    self.dataReceived += 1
                elif self.dataReceived == 1 :
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z

                    self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                    self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)
                    self.dataReceived += 1
                else :
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = arm_pose_tf[0] - self.goalPose.x
                    self.currPos[1] = arm_pose_tf[1] - self.goalPose.y
                    self.currPos[2] = arm_pose_tf[2] - self.goalPose.z

                    self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                    self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)
            else:
                rospy.loginfo('Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()


    def getLearnedParameters(self) :
        logfile = open(self.reproductor_parameters, "r").readlines()
        logfile = [word.strip() for word in logfile]

#        self.Mu_t = numpy.zeros( 3 )

        for i in xrange(len(logfile)) :
            if logfile[i] == 'kV':
                i+=1
                self.kV = float(logfile[i])
            elif logfile[i] == 'kP' :
                i+=1
                self.kP = float(logfile[i])
            elif logfile[i] == 'Mu_t' :
                i+=1
                aux = logfile[i].split(' ')
                self.numStates = len(aux)
                self.Mu_t = numpy.zeros(self.numStates)
                for j in xrange(self.numStates) :
                    self.Mu_t[j] = float(aux[j])
            elif logfile[i] == 'Sigma_t' :
                i+=1
                self.Sigma_t = numpy.zeros(self.numStates)
                for j in xrange(self.numStates) :
                    self.Sigma_t[j] = float(logfile[i])
                    i+=2
            elif logfile[i] == 'Mu_x' :
                i+=1
                self.Mu_x = numpy.zeros(shape=(self.nbVar,self.numStates))
                for k in xrange(self.nbVar) :
                    aux = logfile[i].split(' ')
                    for j in xrange(self.numStates) :
                        self.Mu_x[k,j] = float(aux[j])
                    i+=1
#                rospy.loginfo('Values of the Mu_x \n' + str(self.Mu_x) )
            elif logfile[i] == 'Wp':
                i+=1
                self.Wp = numpy.zeros(shape=(self.numStates,self.nbVar,self.nbVar))
                for z in xrange(self.numStates) :
                    for k in xrange(self.nbVar) :
                        aux = logfile[i].split(' ')
                        for j in xrange(self.nbVar) :
                            self.Wp[z,k,j] = float(aux[j])
                        i+=1
                    i+=1
#                rospy.loginfo('Values of Wp ' + str(self.Wp))
            else :
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
        if numpy.shape(Data) == () :
            nbVar = 1
            nbData = 1
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data - numpy.tile(Mu,(nbData,1))
            prob = (Data*(1/Sigma)) * Data
            prob = math.exp(-0.5*prob) / math.sqrt( numpy.power((2*math.pi),nbVar) * (abs(Sigma)+numpy.finfo(numpy.double).tiny))
            return prob

        else :
            [nbVar,nbData] = numpy.shape(Data)
            #Data = Data' - repmat(Mu',nbData,1);
            Data = Data.T - numpy.tile(Mu.T,(nbData,1))
            prob = numpy.sum( numpy.dot(Data,numpy.linalg.inv(Sigma)) * Data, axis=1)
            prob = math.exp(-0.5*prob) / math.sqrt(  numpy.power((2*math.pi),nbVar) * (abs(numpy.linalg.det(Sigma))+numpy.finfo(numpy.double).tiny))
            return prob
#        Data = Data.T - numpy.tile(Mu.T,(nbData,1))
        #prob = sum((Data*inv(Sigma)).*Data, 2);
#        prob = numpy.sum( numpy.dot(Data,numpy.linalg.inv(Sigma)) * Data, axis=1)
        #realmin = numpy.finfo(numpy.double).tiny
#        prob = math.exp(-0.5*prob) / math.sqrt((2*math.pi)^nbVar * (abs(numpy.linalg.det(Sigma))+numpy.finfo(numpy.double).tiny))


    def loadDemonstration(self) :

        for n in range(len(self.demonstrations)):
            ni=self.demonstrations[n]
            logfile = open(self.demonstration_file+"_"+str(ni)+".csv", "r").readlines()
            pose = numpy.array([[0,0,0]])
            ori = numpy.array([[0,0,0]])
            counter = 0
            traj_demo = Path()
            for line in logfile :
                pose_aux = numpy.array([])
                ori_aux = numpy.array([])
                for word in line.split() :
                    if counter < 3 :
                        pose_aux = numpy.append(pose_aux,word)
                    else :
                        ori_aux = numpy.append(ori_aux,word)
                    counter+=1
                #add positions to the path
                pos_nav = PoseStamped()
                pos_nav.header.frame_id = self.frame_id_goal
                pos_nav.pose.position.x = float(pose_aux[0])
                pos_nav.pose.position.y = float(pose_aux[1])
                pos_nav.pose.position.z = float(pose_aux[2])

                #orientation, is needed a conversion from euler to quaternion
                # pos_nav.pose.point.position.x = pose_aux[0]
                # pos_nav.pose.point.position.y = pose_aux[1]
                # pos_nav.pose.point.position.z = pose_aux[2]

                #add the pose, point to the path
                traj_demo.poses.append(pos_nav)

                #add positions to the matrix
                pose = numpy.vstack((pose,pose_aux))
                ori = numpy.vstack((ori,ori_aux))
                counter = 0

            #publish the message
            traj_demo.header.frame_id = self.frame_id_goal
            self.list_demos[n].publish(traj_demo)

            pose = numpy.vsplit(pose,[1])[1]
            ori = numpy.vsplit(ori,[1])[1]



if __name__ == '__main__':
    try:
        rospy.init_node('learning_reproductor_pose')
        learning_reproductor = learningReproductor( rospy.get_name() )
        learning_reproductor.play()
#        rospy.spin()

    except rospy.ROSInterruptException: pass
