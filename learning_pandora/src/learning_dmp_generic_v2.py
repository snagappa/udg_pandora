#!/usr/bin/env python

"""
This class load the different demonstrations and extract a DMP model to represent
the desired trajectory. The DMP can be configured to use any degrees of freedom,
The last value of the csv file is the time.

@author Arnau Carrera
@date 11/12/2013
"""

# ROS imports
import roslib
roslib.load_manifest('learning_pandora')
import rospy
import math
import numpy as np

from scipy import interpolate
#value to show all the numbers in a matrix
# numpy.set_printoptions(threshold=100000)

#use to load the configuration function
from cola2_lib import cola2_ros_lib

class learningDmp:

    def __init__(self, name):
        """
        This method load the configuration and initialize the different
        variables to be computed. There are not subscribers, publishers or
        tf listener.
        @param name: this atribute contain the name of the rosnode.
        @type name: string
        """
        self.name = name
        self.getConfig()
        rospy.loginfo('Configuration Loaded')
        #Compute the kP and Kv
        # kP = KPmin + (kPmax - kPmin)/2
        # kV = 2*sqrt(kP)

        self.kP = self.kPmin + (self.kPmax - self.kPmin)/2.0
        self.kV = 2.0*np.sqrt(self.kP)

        self.nbSamples = len(self.demonstrations)
        self.d = np.zeros(shape=(self.nbSamples,
                                 self.nbVar*3,
                                 self.nbData))

        self.Data = np.zeros(shape=(self.nbVar*3,
                                    self.nbSamples*self.nbData))
        self.tranning_dt = 0.0

        self.loadDemonstration()

        self.Sigma_x = np.zeros(shape=(self.nbStates,
                                       self.nbVar,
                                       self.nbVar))
        self.Wp = np.zeros(shape=(self.nbStates,
                                  self.nbVar,
                                  self.nbVar))

        self.Mu_t = np.linspace(0, self.nbData*self.tranning_dt, self.nbStates)
        # self.Sigma_t = np.tile((self.nbData*self.dt/self.nbStates)*0.8,
        #                       [self.nbStates, 1, 1])
        self.Sigma_t = np.tile((self.nbData*self.tranning_dt/self.nbStates),
                               [self.nbStates, 1, 1])
        
        rospy.loginfo('Loaded demonstrations')
        # dimensions, trajectory DoF, samples of One Demo

    def getConfig(self):

        param_dict = {'nbData': 'learning/dmp/nbData',
                      'nbDataRepro': 'learning/dmp/nbDataRepro',
                      'nbDataExport': 'learning/dmp/nbDataExport',
                      'nbVar': 'learning/dmp/nbVar',
                      'nbStates': 'learning/dmp/nbStates',
                      'nbDataExport': 'learning/dmp/nbDataExport',
                      'dt': 'learning/dmp/dt',
                      'kPmin': 'learning/dmp/kPmin',
                      'kPmax': 'learning/dmp/kPmax',
                      'alpha': 'learning/dmp/alpha',
                      'demonstration_file': 'learning/dmp/demonstration_file',
                      'demonstrations': 'learning/dmp/demonstrations',
                      'export_filename': 'learning/dmp/export_filename',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.kPmin = np.asarray(self.kPmin)
        self.kPmax = np.asarray(self.kPmax)

    def loadDemonstration(self):
        for n in range(self.nbSamples):
            ni = self.demonstrations[n]
            logfile = open(self.demonstration_file + "_" + str(ni) + ".csv",
                           "r").readlines()
            # vars = np.zeros((1, self.nbVar))
            # Added the time to the var
            vars = np.zeros((1, self.nbVar))
            first_time = -1.0
            last_time = 0.0
            number_data = 0.0
            for line in logfile:
                vars_aux = np.array([])
                for word in line.split():
                    vars_aux = np.append(vars_aux, word)
                if first_time == -1.0:
                    first_time = vars_aux[-1]
                    vars_aux = np.delete(vars_aux, -1)
                else:
                    last_time = vars_aux[-1]
                    vars_aux = np.delete(vars_aux, -1)
                number_data += 1.0
                vars = np.vstack((vars, vars_aux))

            vars = np.vsplit(vars, [1])[1]
            nbDataTmp = vars.shape[0]-1
            #shape(pose,1);
            #xx = linspace(1,nbDataTmp,nbData);
            #d(n).Data(posEndEffectorId,:) = spline(1:nbDataTmp, posEndEffector, xx);
            xx = np.linspace(0, nbDataTmp, self.nbData)
#            rospy.loginfo(str(pose.T[1,:]))
#            rospy.loginfo("range "+ str(range(nbDataTmp)))
            f = interpolate.interp1d(range(nbDataTmp+1), vars.T, kind='cubic')
            yy = f(xx)
            self.d[n, 0:self.nbVar, :] = yy
            #Velocities generated from the interpolation
            #d(n).Data(velId,:) = ([d(n).Data(posId,2:end) d(n).Data(posId,end)]
            #                     - d(n).Data(posId,:)) ./ m.dt;
            aux = np.zeros(shape=(self.nbVar, self.nbData))
            aux[:, 0:-1] = yy[:, 1:]
            aux[:, -1] = yy[:, -1]

            #self.dt = (last_time.astype(np.float) - first_time.astype(np.float)) / number_data
            # rospy.loginfo('Dt ' + str(self.dt))
            self.tranning_dt = (last_time.astype(np.float) - first_time.astype(np.float)) / self.nbData

            self.d[n, self.nbVar:self.nbVar*2, :] = ((aux - yy) / self.dt)

            #Accelerations generated from the interpolation
            #d(n).Data(accId,:) = ([d(n).Data(velId,2:end) d(n).Data(velId,end)]
            #                       - d(n).Data(velId,:)) ./ m.dt;
            aux[:, 0:-1] = self.d[n, self.nbVar:self.nbVar*2, 1:]
            aux[:, -1] = self.d[n, self.nbVar:self.nbVar*2, -1]
            self.d[n, self.nbVar*2:self.nbVar*3, :] = (
                (aux - self.d[n, self.nbVar:self.nbVar*2, :]) / self.dt)
            self.Data[:, ((n)*self.nbData):(self.nbData*(n+1))] = self.d[n, :, :]
            # np.set_printoptions(threshold=100000)
            #rospy.loginfo('\n Values in the d data number ' + str(n) + '\n' +
            #                str(self.d[n,:,:]) + '\n' )
            #            p = raw_input('wait')
            #rospy.loginfo(self.d)

    def trainningDMP(self):
        rospy.loginfo('Learning DMP ...')
        #compute weights
        s = 1
        #Initialization of decay term
        H = np.zeros(shape=(self.nbData, self.nbStates))
        h = np.zeros(shape=(self.nbStates))

        for n in range(self.nbData):
#Update of decay term
            s = s + (-self.alpha*s)*self.tranning_dt
            #s = s + (-self.alpha*s)*self.dt
            t = -math.log(s)/self.alpha
            for i in range(self.nbStates):
#Probability to be in a given state
                h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i, 0, 0])
#Normalization
            H[n, :] = h/np.sum(h)
        #tile equivalent to repmat of matlab
        # Repeat the process for each demonstration
            #rospy.loginfo('Time ' + str(t))
        self.H = np.tile(H, (self.nbSamples, 1))

        #Batch least norm solution to find the centers of the states
#(or primitives) Mu_X (Y=Mu_x*H')
        #         acc                          pos             vel
        #rospy.loginfo('Shape' + str(np.shape(self.Data[self.nbVar*2:self.nbVar*3, :])) + '\nValues of nVar ' + str(self.Data[self.nbVar*2:self.nbVar*3, :]))
        Y = np.zeros(shape=(self.nbVar,
                            self.nbSamples*self.nbData))
        for i in range(self.nbVar):
            Y[i,:] = (self.Data[(i+self.nbVar*2), :]*(1/self.kP[i]) +
                      self.Data[i, :] +
                      self.Data[(i+self.nbVar), :]*(self.kV[i]/self.kP[i]))

        # Y = (self.Data[self.nbVar*2:self.nbVar*3, :]*(1/self.kP) +
        #      self.Data[0:self.nbVar, :] +
        #      self.Data[self.nbVar:self.nbVar*2, :]*(self.kV/self.kP))
#        rospy.loginfo(' Pseudo inversa H.t \n ' +
#                      str( np.linalg.pinv(self.H.T) ) + '\n' )
#Pseudoinverse solution Mu_x = [inv(H'*H)*H'*Y']'
        self.Mu_x = np.dot(Y, np.linalg.pinv(self.H.T))

# Mu_x and H are equal to matlab.
# Y seem equal but Its difficult to see too many values

        #Compute residuals
#Regularization term for matrix inversion
        RI = np.eye(self.nbVar, self.nbVar)*1E-3

#        rospy.loginfo('Mu_x \n' + str(self.Mu_x) + '\n' )
#        casa = raw_input('Check Mu_x')
#        rospy.loginfo('H \n' + str(self.H) + '\n' )
#        casa = raw_input('Check H')
#        np.set_printoptions(threshold=100000)
#        rospy.loginfo('Values of Y \n' + str(Y) + '\n' )
#        casa = raw_input('Check Reshape')

        for i in range(self.nbStates):
            a = Y-np.tile(self.Mu_x[:, i].reshape(len(self.Mu_x[:, i]), 1),
                          (1, self.nbData*self.nbSamples))
            b = np.diag(self.H[:, i])
            product = np.dot(a, b)
            self.Sigma_x[i, :, :] = np.cov(product)
#Use variation information to determine stiffness
            self.Wp[i, :, :] = np.linalg.inv(self.Sigma_x[i, :, :]+RI)
        # Warning sigmas are different from the matlab results I don

#        rospy.loginfo('Values of Sigma_x \n ' + str(self.Sigma_x) + '\n')

        #Rescale Wp to stay within the [kPmin,kPmax] range
        V = np.zeros(shape=(self.nbStates, self.nbVar, self.nbVar))
        lambda_var = np.zeros(shape=(self.nbVar, self.nbStates))
#        rospy.loginfo( 'Values of Wp \n' + str(self.Wp) + '\n' )
        for i in range(self.nbStates):
#Eigencomponents decomposition
            [Dtmp, V[i, :, :]] = np.linalg.eig(self.Wp[i, :, :])
            lambda_var[:, i] = Dtmp
            # The eigen values Dtmp can be different from the matlab answers

#        rospy.loginfo( 'Values of V \n' + str(V) +'\n' )
        lambda_min = np.min(lambda_var)
        lambda_max = np.max(lambda_var)

        for i in range(self.nbStates):
        #Full covariance stiffness matrix derived from residuals estimation
        #Rescale each eigenvalue such that they lie in the range [kPmin,kPmax]
            Dtmp = np.diag((self.kPmax-self.kPmin) *
                           (lambda_var[:, i]-lambda_min) /
                           (lambda_max-lambda_min) + self.kPmin)
#Reconstruction from the modified eigencomponents
            # self.Wp[i, :, :] = np.dot(V[i, :, :],
            #                           np.dot(Dtmp, np.linalg.inv(V[i, :, :])))
            self.Wp[i, :, :] = np.dot(V[i, :, :],
                                      np.dot(Dtmp, np.linalg.pinv(V[i, :, :])))
        #OR
            #Standard DMP
            #self.Wp[:,:,i]=np.diag(np.ones(shape=(self.nbVar,1))*self.kP)

        #rospy.loginfo('\nValues Wp \n ' + str(self.Wp) + '\n' )

        rospy.loginfo('Learning finished successfully')

        self.exportPlayData()

    def gaussPDF(self, Data, Mu, Sigma):
###     This function computes the Probability Density Function (PDF) of a
###     multivariate Gaussian represented by means and covariance matrix.
###
###     Author:	Sylvain Calinon, 2009
###             http://programming-by-demonstration.org
###
###     Inputs ---------------------------------------------------------------
###      o Data:  D x N array representing N datapoints of D dimensions.
###      o Mu:    D x K array representing the centers of the K GMM components.
###      o Sigma: D x D x K array representing the covariance matrices of the
###                  K GMM components.
###     Outputs --------------------------------------------------------------
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
#prob = math.exp(-0.5*prob) / math.sqrt((2*math.pi)^nbVar *
#(abs(np.linalg.det(Sigma))+np.finfo(np.double).tiny))

    def exportPlayData(self):
    #EXPORTPLAYDATA
    # This function export the data to be used with the prepare ekf_Track
    # source code.
        file = open(self.export_filename, 'w')

        file.write('kV\n')
        for j in self.kV:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('kP\n')
        for j in self.kP:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('Mu_t\n')
        for j in self.Mu_t:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('Sigma_t\n')
        for i in range(self.Sigma_t.shape[0]):
            for k in range(self.Sigma_t.shape[2]):
                for j in self.Sigma_t[i, k, :]:
                    file.write(str(j)+' ')
                file.write('\n')
            file.write('\n')

        file.write('Mu_x\n')

        for i in range(self.Mu_x.shape[0]):
            for j in self.Mu_x[i, :]:
                file.write(str(j)+' ')
            file.write('\n')

        file.write('\n')
        file.write('Wp\n')

        for i in range(self.Wp.shape[0]):
            for k in range(self.Wp.shape[2]):
                for j in self.Wp[i, :, k]:
                    file.write(str(j)+' ')
                file.write('\n')
            file.write('\n')

        file.close()
        rospy.loginfo('The parameters learned has been exported to '
                      + self.export_filename)

if __name__ == '__main__':
    try:
        rospy.init_node('learning_dmp_v2')
        learning_dmp = learningDmp(rospy.get_name())
        learning_dmp.trainningDMP()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
