#!/usr/bin/env python

import numpy as np

class LearningDmpGeneric(object):
    """
    This class contains the generic dmp. Capable to load the files and learn
    from one time to the end of another time
    """

    def __init__(self, kP, kV, kP_min, kP_max, alpha, states, dof, nb_data,
                 file_name, samples, init_time, end_time, output_file_name):
        """
        Initialize the class
        """
        self.kP = kP
        self.kV = kV
        self.kP_min = kP_min
        self.kP_max = kP_max
        self.nb_data = nb_data
        self.alpha = alpha
        self.states = states
        self.dof = dof
        self.file_name = file_name
        self.samples = samples
        self.init_time = init_time
        self.end_time = end_time
        self.output_file_name = output_file_name
        self.nb_samples = len(samples)

        self.d = np.zeros(shape=(self.nb_samples,
                                 self.dof*3,
                                 self.nb_data))

        self.Data = np.zeros(shape=(self.dof*3,
                                    self.nb_samples*self.nb_data))
        self.avg_dt = 0.0

        self.loadDemonstration()

        self.sigma_x = np.zeros(shape=(self.states,
                                       self.dof,
                                       self.dof))
        self.wp = np.zeros(shape=(self.states,
                                  self.dof,
                                  self.dof))
        print 'Loaded demonstrations'

    def loadDemonstration(self):
        """
        Load Demonstrations from the last point to the begining
        """
        for n in range(self.nb_samples):
            ni = self.samples[n]
            logfile = open(self.file_name + "_" + str(ni) + ".csv",
                           "r").readlines()
            # vars = np.zeros((1, self.nbVar))
            # Added the time to the var
            vars = np.zeros((1, self.dof))
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
            xx = np.linspace(0, nbDataTmp, self.nb_data)
#            rospy.loginfo(str(pose.T[1,:]))
#            rospy.loginfo("range "+ str(range(nbDataTmp)))
            f = interpolate.interp1d(range(nbDataTmp+1), vars.T, kind='cubic')
            yy = f(xx)
            self.d[n, 0:self.dof, :] = yy
            #Velocities generated from the interpolation
            #d(n).Data(velId,:) = ([d(n).Data(posId,2:end) d(n).Data(posId,end)]
            #                     - d(n).Data(posId,:)) ./ m.dt;
            aux = np.zeros(shape=(self.dof, self.nb_data))
            aux[:, 0:-1] = yy[:, 1:]
            aux[:, -1] = yy[:, -1]

            #self.dt = (last_time.astype(np.float) - first_time.astype(np.float)) / number_data
            # rospy.loginfo('Dt ' + str(self.dt))
            tranning_dt = (last_time.astype(np.float)
                           - first_time.astype(np.float)) / self.nb_data

            #self.d[n, self.nbVar:self.nbVar*2, :] = ((aux - yy) / self.dt)
            self.d[n, self.dof:self.dof*2, :] = ((aux - yy) / tranning_dt)

            #Accelerations generated from the interpolation
            #d(n).Data(accId,:) = ([d(n).Data(velId,2:end) d(n).Data(velId,end)]
            #                       - d(n).Data(velId,:)) ./ m.dt;
            aux[:, 0:-1] = self.d[n, self.dof:self.dof*2, 1:]
            aux[:, -1] = self.d[n, self.dof:self.dof*2, -1]
            self.d[n, self.dof*2:self.dof*3, :] = (
                (aux - self.d[n, self.dof:self.dof*2, :]) / tranning_dt)
                #(aux - self.d[n, self.nbVar:self.nbVar*2, :]) / self.dt)
            self.Data[:, ((n)*self.nb_data):(self.nb_data*(n+1))] = self.d[n, :, :]
            # np.set_printoptions(threshold=100000)
            #rospy.loginfo('\n Values in the d data number ' + str(n) + '\n' +
            #                str(self.d[n,:,:]) + '\n' )
            #            p = raw_input('wait')
            #rospy.loginfo(self.d)
            self.avg_dt += tranning_dt
        self.avg_dt = self.avg_dt/self.nb_samples
        self.mu_t = np.linspace(0, self.nb_data*self.avg_dt, self.states)
        # self.Sigma_t = np.tile((self.nbData*self.dt/self.nbStates)*0.8,
        #                       [self.nbStates, 1, 1])
        self.sigma_t = np.tile((self.nb_data*self.avg_dt/self.states),
                               [self.states, 1, 1])

    def trainningDMP(self):
        rospy.loginfo('Learning DMP ...')
        #compute weights
        s = 1
        #Initialization of decay term
        H = np.zeros(shape=(self.nb_data, self.states))
        h = np.zeros(shape=(self.states))

        for n in range(self.nb_data):
#Update of decay term
            s = s + (-self.alpha*s)*self.avg_dt
            #s = s + (-self.alpha*s)*self.dt
            t = -math.log(s)/self.alpha
            for i in range(self.states):
#Probability to be in a given state
                h[i] = self.gaussPDF(t, self.mu_t[i], self.sigma_t[i, 0, 0])
#Normalization
            H[n, :] = h/np.sum(h)
        rospy.loginfo('Avg Time ' + str(self.avg_dt*self.nb_data))
        #tile equivalent to repmat of matlab
        # Repeat the process for each demonstration
            #rospy.loginfo('Time ' + str(t))
        self.H = np.tile(H, (self.nb_samples, 1))

        #Batch least norm solution to find the centers of the states
#(or primitives) Mu_X (Y=Mu_x*H')
        #         acc                          pos             vel
        #rospy.loginfo('Shape' + str(np.shape(self.Data[self.nb_var*2:self.nb_var*3, :])) + '\nValues of nVar ' + str(self.Data[self.nb_var*2:self.nb_var*3, :]))
        Y = np.zeros(shape=(self.nb_var,
                            self.nb_samples*self.nb_data))
        for i in range(self.nb_var):
            Y[i,:] = (self.Data[(i+self.dof*2), :]*(1/self.kP[i]) +
                      self.Data[i, :] +
                      self.Data[(i+self.dof), :]*(self.kV[i]/self.kP[i]))

        # Y = (self.Data[self.nb_var*2:self.nb_var*3, :]*(1/self.kP) +
        #      self.Data[0:self.nb_var, :] +
        #      self.Data[self.nb_var:self.nb_var*2, :]*(self.kV/self.kP))
#        rospy.loginfo(' Pseudo inversa H.t \n ' +
#                      str( np.linalg.pinv(self.H.T) ) + '\n' )
#Pseudoinverse solution Mu_x = [inv(H'*H)*H'*Y']'
# The H is transpossed to match the Y values for each time stamp
        self.mu_x = np.dot(Y, np.linalg.pinv(self.H.T))

# Mu_x and H are equal to matlab.
# Y seem equal but Its difficult to see too many values

        #Compute residuals
#Regularization term for matrix inversion
        RI = np.eye(self.dof, self.dof)*1E-3

#        rospy.loginfo('Mu_x \n' + str(self.Mu_x) + '\n' )
#        casa = raw_input('Check Mu_x')
#        rospy.loginfo('H \n' + str(self.H) + '\n' )
#        casa = raw_input('Check H')
#        np.set_printoptions(threshold=100000)
#        rospy.loginfo('Values of Y \n' + str(Y) + '\n' )
#        casa = raw_input('Check Reshape')

        for i in range(self.states):
            a = Y-np.tile(self.mu_x[:, i].reshape(len(self.mu_x[:, i]), 1),
                          (1, self.nb_data*self.nb_samples))
            b = np.diag(self.H[:, i])
            product = np.dot(a, b)
            self.sigma_x[i, :, :] = np.cov(product)
#Use variation information to determine stiffness
            self.wp[i, :, :] = np.linalg.pinv(self.sigma_x[i, :, :]+RI)
        # Warning sigmas are different from the matlab results I don

#        rospy.loginfo('Values of Sigma_x \n ' + str(self.Sigma_x) + '\n')

        #Rescale Wp to stay within the [kPmin,kPmax] range
        V = np.zeros(shape=(self.states, self.dof, self.nb_var))
        lambda_var = np.zeros(shape=(self.dof, self.states))
#        rospy.loginfo( 'Values of Wp \n' + str(self.Wp) + '\n' )
        for i in range(self.states):
#Eigencomponents decomposition
            [Dtmp, V[i, :, :]] = np.linalg.eig(self.wp[i, :, :])
            lambda_var[:, i] = Dtmp
            # The eigen values Dtmp can be different from the matlab answers

#        rospy.loginfo( 'Values of V \n' + str(V) +'\n' )
        lambda_min = np.min(lambda_var)
        lambda_max = np.max(lambda_var)

        for i in range(self.states):
        #Full covariance stiffness matrix derived from residuals estimation
        #Rescale each eigenvalue such that they lie in the range [kPmin,kPmax]
            Dtmp = np.diag((self.kP_max-self.kP_min) *
                           (lambda_var[:, i]-lambda_min) /
                           (lambda_max-lambda_min) + self.kP_min)
#Reconstruction from the modified eigencomponents
            # self.Wp[i, :, :] = np.dot(V[i, :, :],
            #                           np.dot(Dtmp, np.linalg.inv(V[i, :, :])))
            self.wp[i, :, :] = np.dot(V[i, :, :],
                                      np.dot(Dtmp, np.linalg.pinv(V[i, :, :])))
        #OR
            #Standard DMP
            #self.Wp[:,:,i]=np.diag(np.ones(shape=(self.nb_var,1))*self.kP)

        #rospy.loginfo('\nValues Wp \n ' + str(self.Wp) + '\n' )

        rospy.loginfo('Learning finished successfully')

    def exportPlayData(self):
        """
        Export model in a file
        """
        file = open(self.output_file_name, 'w')

        file.write('kV\n')
        for j in self.kV:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('kP\n')
        for j in self.kP:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('Mu_t\n')
        for j in self.mu_t:
            file.write(str(j)+' ')
        file.write('\n\n')

        file.write('Sigma_t\n')
        for i in range(self.sigma_t.shape[0]):
            for k in range(self.sigma_t.shape[2]):
                for j in self.sigma_t[i, k, :]:
                    file.write(str(j)+' ')
                file.write('\n')
            file.write('\n')

        file.write('mu_x\n')

        for i in range(self.mu_x.shape[0]):
            for j in self.mu_x[i, :]:
                file.write(str(j)+' ')
            file.write('\n')

        file.write('\n')
        file.write('Wp\n')

        for i in range(self.wp.shape[0]):
            for k in range(self.wp.shape[2]):
                for j in self.wp[i, :, k]:
                    file.write(str(j)+' ')
                file.write('\n')
            file.write('\n')

        file.close()
        rospy.loginfo('The parameters learned has been exported to '
                      + self.output_file_name)

    def __del__(self):
        print 'Destroy everything'
