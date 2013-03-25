#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#use to load the configuration function
import cola2_ros_lib

#use to normalize the angle
import cola2_lib

# import the message to know the position
from geometry_msgs.msg import PoseStamped

#include message of the point
from sensor_msgs.msg import Joy

import math
import numpy

import threading
import tf

class learningReproductor :

    def __init__(self , name) :
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.goalPose = PoseStamped()
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
        self.pub_desired_position = rospy.Publisher("cola2_control/joystick_arm_data", Joy )

        rospy.Subscriber("/arm/valve_pose", PoseStamped, self.updateGoalPose)
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose )

        rospy.loginfo('Configuration ' + str(name) +  ' Loaded ')


    def getConfig(self) :
        param_dict = {'reproductor_parameters': 'learning/reproductor/parameters',
                      'alpha': 'learning/reproductor/alpha',
                      's': 'learning/reproductor/s',
                      'nbVar': 'learning/reproductor/nbVar',
                      'interval_time': 'learning/reproductor/interval_time',
                      'simulation': 'learning/reproductor/simulation',
                      'nbDataRepro': 'learning/reproductor/nbDataRepro',
                      'exportFile': 'learning/reproductor/exportFile'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Interval time value: ' + str(self.interval_time) )

#WARNING THIS HAS NOT SENSE
# THE UPDATE WILL HAVE TO BE UPDATED BY THE DETECTION NOT FROM THE UPDATE
    def updateGoalPose(self, goalPose):
        self.lock.acquire()
        try:
            self.goalPose = goalPose
            self.dataGoalReceived = True
        finally:
            self.lock.release()

    def updateArmPose (self, armPose):
        self.lock.acquire()
        try:
            if self.dataGoalReceived :
                if self.dataReceived == 0 :
                    self.currPos[0] = self.goalPose.pose.position.x - armPose.pose.position.x
                    self.currPos[1] = self.goalPose.pose.position.y - armPose.pose.position.y
                    self.currPos[2] = self.goalPose.pose.position.z - armPose.pose.position.z

                    self.currTime = armPose.header.stamp.secs + (armPose.header.stamp.nsecs*1E-9)
                    self.dataReceived += 1


                elif self.dataReceived == 1 :
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = self.goalPose.pose.position.x - armPose.pose.position.x
                    self.currPos[1] = self.goalPose.pose.position.y - armPose.pose.position.y
                    self.currPos[2] = self.goalPose.pose.position.z - armPose.pose.position.z

                    self.currTime = armPose.header.stamp.secs + (armPose.header.stamp.nsecs*1E-9)
                    self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)

                    self.dataReceived += 1
                else :
                    self.prevPos = self.currPos
                    self.prevTime = self.currTime

                    self.currPos[0] = self.goalPose.pose.position.x - armPose.pose.position.x
                    self.currPos[1] = self.goalPose.pose.position.y - armPose.pose.position.y
                    self.currPos[2] = self.goalPose.pose.position.z - armPose.pose.position.z

                    self.currTime = armPose.header.stamp.secs + (armPose.header.stamp.nsecs*1E-9)
                    self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)

            else :
                rospy.loginfo('Waiting to initialise the valve and robot position')
        finally:
            self.lock.release()

    def play(self) :
#        pub = rospy.Publisher('arm', )
        while not rospy.is_shutdown():
            self.lock.acquire()
            try:
                if not self.simulation :
                    if  self.dataReceived >1 :
                        rospy.loginfo('Data received')
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

        #rospy.loginfo( 'CurrWp \n' + currWp )
        #rospy.loginfo( 'CurrWp \n' + currWp )
        self.currAcc = (numpy.dot(currWp, (currTar-self.currPosSim))) - (self.kV*self.currVel)

        self.currVel = self.currVel + (self.currAcc * self.interval_time)
        self.desPos = self.currPosSim + (self.currVel * self.interval_time)

        s = repr( self.desPos[0] ) + " " + repr( self.desPos[1]) +  " " + repr(self.desPos[2]) + "\n"
        self.file.write(s)

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

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
        #NOT needed
        self.desPos = self.currPos + self.desVel * self.interval_time

        #s = repr( self.currPos[0] ) + " " + repr( self.currPos[1]) +  " " + repr(self.currPos[2]) + "\n"
        #self.fileTraj.write(s)
        self.publishArmPose()

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time


    def publishArmPose(self) :
        joyCommand = Joy()

#        rospy.loginfo('Goal pose' + str(self.goalPose.pose.position.x))
#        rospy.loginfo('Des pose' + str(self.desPos))

        newArmPose_x = self.goalPose.pose.position.x + self.desPos[0] # desired_pose_tf[0]
        newArmPose_y = self.goalPose.pose.position.y + self.desPos[1] # desired_pose_tf[1]
        newArmPose_z = self.goalPose.pose.position.z + self.desPos[2] # desired_pose_tf[2]

#        rospy.loginfo('Arm pose' + str(self.armPose.pose.position.x))

        command_x = -(self.desPos[0] - self.currPos[0])
        command_y = -(self.desPos[1] - self.currPos[1])
        command_z = -(self.desPos[2] - self.currPos[2])

        # command_x = newArmPose_x - self.armPose.pose.position.x
        # command_y = newArmPose_y - self.armPose.pose.position.y
        # command_z = newArmPose_z - self.armPose.pose.position.z

        joyCommand.axes.append( command_x )
        joyCommand.axes.append( command_y )
        joyCommand.axes.append( command_z )
        joyCommand.axes.append( 0.0 )
        joyCommand.axes.append( 0.0 )
        joyCommand.axes.append( 0.0 )

        s = repr( self.currPos[0] ) + " " + repr( self.currPos[1]) +  " " + repr(self.currPos[2]) + "\n"
        self.fileTraj.write(s)

        self.pub_desired_position.publish(joyCommand)


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


if __name__ == '__main__':
    try:
        rospy.init_node('learning_reproductor')
        learning_reproductor = learningReproductor( rospy.get_name() )
        learning_reproductor.play()
#        rospy.spin()

    except rospy.ROSInterruptException: pass
