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


import math
import numpy
from scipy import interpolate

import threading

#value to show all the numbers in a matrix
# numpy.set_printoptions(threshold=100000)

class learningReproductor :

    def __init__(self , name) :
        self.name = name
        self.getConfig()
        self.getLearnedParameters()
        self.prevPos = numpy.zeros(self.dimensions)
        self.prevTime = 0.0
        self.currTime = 0.0
        self.currPos = numpy.zeros(self.dimensions)
        self.currVel = numpy.zeros(self.dimensions)
        self.currAcc = numpy.zeros(self.dimensions)
        self.dataReceived = 0 
        self.lock = threading.Lock()
        self.pub_desired_position = rospy.Publisher("/arm/desired_position", PoseStamped )
        
        rospy.Subscriber('/arm/pose_stamped', PoseStamped , self.updateArmPosition )

        rospy.loginfo('Configuration ' str(name)   ' Loaded ')


    def getConfig(self) :
        param_dict = {'reproductor_parameters': 'learning/reproductor/parameters',
                      'alpha': 'learning/reproductor/alpha', 
                      's': 'learning/reproductor/s',
                      'interval_time': 'learning/reproductor/interval_time'}
        cola2_ros_lib.getRosParams(self, param_dict)
            
    def play(self) :
#        pub = rospy.Publisher('arm', )
        while not rospy.is_shutdown():
            self.lock.acquire()
            try:
                self.generateNewPose()
            finally:
                self.lock.release()
            rospy.sleep(self.interval_time)



    def generateNewPose(self) :
        t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        h = numpy.zeros(self.numStates) 
        for i in xrange(self.numStates) :
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])
        # normalize the value
        h = h / numpy.sum(h)
        
        #init to vectors
        currTar = numpy.zeros(self.dimensions)
        currWp = numpy.zeros(shape=(self.dimensions,self.dimensions))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.numStates ) :
            currTar = currTar + self.Mu_x[:,i]*h[i]
            currWp = currWp + self.Wp[i,:,:]*h[i]

        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);
        self.currAcc = (currWp * (currTar-self.currPos)) - (self.kV*self.currVel)
        # action is a scalar value to evaluate the safety
        #currAcc = currAcc * math.fabs(self.action)
        
        self.currVel = self.currVel + self.currAcc * self.interval_time
        self.desPos = self.currPos + self.currVel * self.interval_time

        #self.s = self.s + (-self.alpha*self.s)*self.interval_time*self.action
        self.s = self.s + (-self.alpha*self.s)*self.interval_time

    def updateArmPosition(self, data):
        self.lock.acquire()
        try:
            if self.dataReceived == 0 :
                self.currPos[0] = data.pose.position.x                
                self.currPos[1] = data.pose.position.y
                self.currPos[2] = data.pose.position.z
                self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                self.dataReceived += 1
            elif self.dataReceived == 1 :
                self.prevPos = self.currPos
                self.prevTime = self.currTime
                self.currPos[0] = data.pose.position.x                
                self.currPos[1] = data.pose.position.y
                self.currPos[2] = data.pose.position.z
                self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)
                self.dataReceived += 1
            else :
                self.prevPos = self.currPos
                self.prevTime = self.currTime
                self.currPos[0] = data.pose.position.x                
                self.currPos[1] = data.pose.position.y
                self.currPos[2] = data.pose.position.z
                self.currTime = data.header.stamp.secs + (data.header.stamp.nsecs*1E-9)
                self.currVel = (self.currPos-self.prevPos) / (self.currTime-self.prevTime)         
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
                self.dimensions = len(logfile[i].split(' '))
                self.Mu_x = numpy.zeros(shape=(self.dimensions, self.numStates))
                for k in xrange(self.numStates) :
                    aux = logfile[i].split(' ')
                    for j in xrange(self.dimensions) :
                        self.Mu_x[k,j] = float(aux[j])
                    i+=1
#                rospy.loginfo('Values of the Mu_x \n' + str(self.Mu_x) )
            elif logfile[i] == 'Wp':
                i+=1
                self.Wp = numpy.zeros(shape=(self.numStates,self.dimensions,self.dimensions))
                for z in xrange(self.numStates) :
                    for k in xrange(self.numStates) :
                        aux = logfile[i].split(' ')
                        for j in xrange(self.dimensions) :
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
