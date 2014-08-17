#!/usr/bin/env python

import numpy as np
import math

#use to normalize the angle
# ULLL
from cola2_lib import cola2_lib

class LearningDmpReproductor(object):

    def __init__(self, name, file_name, dof, alpha, interval_time ):
        self.name = name
        self.file_name = file_name
        self.interval_time = interval_time
        self.dof = dof
        self.states = 0
        self.kV = 0
        self.kP = 0
        self.Mu_t = 0
        self.Sigma_t = 0
        self.Mu_x = 0
        self.Wp = 0
        self.get_learned_parameters()
        self.s = 1.0
        self.alpha = alpha
        self.action = 1.0

    def get_learned_parameters(self):
        """
        This method loads the data from a .txt file generated by the dmp
        learning.
        """
        #read the file
        logfile = open(self.file_name, "r").readlines()

        logfile = [word.strip() for word in logfile]
        for i in xrange(len(logfile)):
            if logfile[i] == 'kV':
                i += 1
                # Individual KV
                self.kV = float(logfile[i])
            elif logfile[i] == 'kP':
                i += 1
                # Individual KV
                self.kP = float(logfile[i])
            elif logfile[i] == 'Mu_t':
                i += 1
                aux = logfile[i].split(' ')
                self.states = len(aux)
                self.Mu_t = np.zeros(self.states)
                for j in xrange(self.states):
                    self.Mu_t[j] = float(aux[j])
            elif logfile[i] == 'Sigma_t':
                i += 1
                self.Sigma_t = np.zeros(self.states)
                for j in xrange(self.states):
                    self.Sigma_t[j] = float(logfile[i])
                    i += 2
            elif logfile[i] == 'mu_x':
                i += 1
                self.Mu_x = np.zeros(shape=(self.dof, self.states))
                for k in xrange(self.dof):
                    aux = logfile[i].split(' ')
                    for j in xrange(self.states):
                        self.Mu_x[k, j] = float(aux[j])
                    i += 1
            elif logfile[i] == 'Wp':
                i += 1
                self.Wp = np.zeros(
                    shape=(self.states, self.dof, self.dof))
                for z in xrange(self.states):
                    for k in xrange(self.dof):
                        aux = logfile[i].split(' ')
                        for j in xrange(self.dof):
                            self.Wp[z, k, j] = float(aux[j])
                        i += 1
                    i += 1
            else:
                pass

    def generateNewPose(self, current_pose, current_vel):
        """
        Generates the new position in the current state
        """
        t = -math.log(self.s)/self.alpha
        #self.tf
        # if self.backward :
        #     t = self.tf + math.log(self.s)/self.alpha
        # else :
        #     t = -math.log(self.s)/self.alpha
        # for each atractor or state obtain the weigh
        #rospy.loginfo('Time :' + str(t) )
        h = np.zeros(self.states)
        for i in xrange(self.states):
            h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i])

        # normalize the value
        if t > self.Mu_t[self.states-1]+(self.Sigma_t[self.states-1]*1.2):
            print 'The time used in the demonstration is exhausted'
            #self.enabled = False
            self.s = 1.0
            return [[],[]]
        else:
            h = h / np.sum(h)

        #init to vectors
        currTar = np.zeros(self.dof)
        currWp = np.zeros(shape=(self.dof, self.dof))

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        for i in xrange(self.dof):
            currTar = currTar + self.Mu_x[:, i]*h[i]
            currWp = currWp + self.Wp[i, :, :]*h[i]

        #rospy.loginfo('Current Tar '+ str(currTar))
        #rospy.loginfo('Current Wp '+ str(currWp))
        #Compute acceleration
        #currAcc = currWp * (currTar-currPos) - ( m.kV * currVel);

        diff = currTar-current_pose
        diff[3] = cola2_lib.normalizeAngle(diff[3])

        #rospy.loginfo('Kv ' + str(self.kV.tolist()))
        desAcc = np.dot(
            currWp, diff) - np.dot(self.kV, current_vel)
        # action is a scalar value to evaluate the safety
        #rospy.loginfo('Des Acc' + str(self.desAcc))

        desVel = current_vel + desAcc * self.interval_time
        #NOT needed
        desPos = current_pose + desVel * self.interval_time

        self.s = self.s - self.alpha*self.s*self.interval_time*self.action#*1.5

        return [desPos, desVel]

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

    def __del__(self):
        print 'Deleted the Dmp call ' + self.file_name
