#!/usr/bin/env python

import numpy as np
import math

#use to normalize the angle
# ULLL
from cola2_lib import cola2_lib

class LearningDmpParamReproductor(object):

    def __init__(self, file_name, file_directory, dof, alpha, interval_time, nb_groups ):
        #self.file_name = name
        self.file_name = file_directory+'/'+file_name
        self.interval_time = interval_time
        self.dof = dof
        self.nb_groups = nb_groups
        self.states = 0
        self.kV = [0]*self.nb_groups
        self.kP = [0]*self.nb_groups
        self.Mu_t = [None]*self.nb_groups
        self.Sigma_t = [None]*self.nb_groups
        self.Mu_x = [None]*self.nb_groups
        self.Wp = [None]*self.nb_groups
        self.dofs = [None]*self.nb_groups
        self.value_group = [0]*self.nb_groups
        self.get_learned_parameters()
        self.s = 1.0
        self.alpha = alpha
        self.action = 1.0

    def get_learned_parameters(self):
        for n in range(self.nb_groups):
            self.get_learned_parameters_group(n)


    def get_learned_parameters_group(self, group):
        """
        This method loads the data from a .txt file generated by the dmp
        learning.
        """
        #read the file
        logfile = open(self.file_name+'_'+str(group)+'.txt', "r").readlines()

        logfile = [word.strip() for word in logfile]
        for i in xrange(len(logfile)):
            if logfile[i] == 'kV':
                i += 1
                # Individual KV
                self.kV[group] = float(logfile[i])
            elif logfile[i] == 'kP':
                i += 1
                # Individual KV
                self.kP[group] = float(logfile[i])
            elif logfile[i] == 'Mu_t':
                i += 1
                aux = logfile[i].split(' ')
                self.states = len(aux)
                self.Mu_t[group] = np.zeros(self.states)
                for j in xrange(self.states):
                    self.Mu_t[group][j] = float(aux[j])
            elif logfile[i] == 'Sigma_t':
                i += 1
                self.Sigma_t[group] = np.zeros(self.states)
                for j in xrange(self.states):
                    self.Sigma_t[group][j] = float(logfile[i])
                    i += 2
            elif logfile[i] == 'mu_x':
                i += 1
                self.Mu_x[group] = np.zeros(shape=(self.dof, self.states))
                for k in xrange(self.dof):
                    aux = logfile[i].split(' ')
                    for j in xrange(self.states):
                        self.Mu_x[group][k, j] = float(aux[j])
                    i += 1
            elif logfile[i] == 'Wp':
                i += 1
                self.Wp[group] = np.zeros(
                    shape=(self.states, self.dof, self.dof))
                for z in xrange(self.states):
                    for k in xrange(self.dof):
                        aux = logfile[i].split(' ')
                        for j in xrange(self.dof):
                            self.Wp[group][z, k, j] = float(aux[j])
                        i += 1
                    i += 1
            elif logfile[i] == 'Dofs':
                i += 1
                aux = logfile[i].split(' ')
                length = len(aux)
                self.dofs[group] = np.zeros(length, dtype=np.int32)
                for j in xrange(length):
                    self.dofs[group][j] = int(aux[j])
            elif logfile[i] == 'ParamValue':
                i += 1
                self.value_group[group] = float(logfile[i])
            else:
                pass

    def generateNewPose(self, current_pose, current_vel, action, param):
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
        h = [np.zeros(self.states)]*self.nb_groups
        for j in xrange(self.nb_groups):
            for i in xrange(self.states):
                h[j][i] = self.gaussPDF(t, self.Mu_t[j][i], self.Sigma_t[j][i])

            # normalize the value
            if t > self.Mu_t[j][self.states-1]+(self.Sigma_t[j][self.states-1]*1.2):
                print 'The time used in the demonstration is exhausted'
                #self.enabled = False
                self.s = 1.0
                return [[],[]]
            else:
                h[j] = h[j] / np.sum(h[j])

        # Generate parameters influence
        # First we compute the distance between the parameter and the demonstrations
        differences = np.abs(np.array(self.value_group[:]) - param)
        # we search for the limits
        #max_param = differences.max()
        #min_param = differences.min()
        total = differences.sum()
        influence = 1- (differences / total)
        influence = influence / influence.sum()

        #init to vectors
        currTar = [np.zeros(self.dof)]*self.nb_groups
        currWp = [np.zeros(shape=(self.dof, self.dof))]*self.nb_groups

        #For each actuator, State, Acumulate the position using weigh
        #CurrTar = The center of the GMM * weight of the state
        #CurrWp = Sigma of the GMM * weight of the State

        param_tar = np.zeros(self.dof)
        param_wp = np.zeros(shape=(self.dof, self.dof))
        param_vel = np.zeros(self.dof)
        for j in xrange(self.nb_groups):
            for i in xrange(self.states):
                currTar[j] = currTar[j] + self.Mu_x[j][:, i]*h[j][i]
                currWp[j] = currWp[j] + self.Wp[j][i, :, :]*h[j][i]
            param_tar = param_tar + currTar[j]*influence[j]
            param_wp = param_wp + currWp[j]*influence[j]
            param_vel = param_vel + self.kV[j]*influence[j]
        #rospy.loginfo('Current Tar '+ str(currTar))
        #rospy.loginfo('Current Wp '+ str(currWp))
        #Compute acceleration
        #currAccf = currWp * (currTar-currPos) - ( m.kV * currVel);

        #Current pose has to bee a np array
        #Work Around to aboid the diference size of the algorithm
        #print 'Size pose ' + str(current_pose) + ' Size dof ' + str(self.dof)
        if len(current_pose) != self.dof:
            selected_pose = current_pose[self.dofs == 1]
            selected_vel = current_vel[self.dofs == 1]
        else:
            selected_pose = current_pose
            selected_vel = current_vel
        diff = param_tar-selected_pose
        #diff[3] = cola2_lib.normalizeAngle(diff[3])

        #rospy.loginfo('Kv ' + str(self.kV.tolist()))
        desAcc = np.dot(
            param_wp, diff) - np.dot(param_vel, selected_vel)
        # action is a scalar value to evaluate the safety
        #rospy.loginfo('Des Acc' + str(self.desAcc))

        desVel = selected_vel + desAcc * self.interval_time
        #NOT needed
        desPos = selected_pose + desVel * self.interval_time

        self.s = self.s - self.alpha*self.s*self.interval_time*action#*1.5

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
