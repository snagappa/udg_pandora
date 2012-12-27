#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import math
import numpy

from scipy import interpolate
#value to show all the numbers in a matrix
# numpy.set_printoptions(threshold=100000)

class learningDmp :
    
    def __init__(self, name):
        self.name = name
        self.getConfig()
        rospy.loginfo('Configuration Loaded')
        self.nbSamples = len(self.demonstrations)
        self.d = numpy.zeros(shape=(self.nbSamples,9,self.nbData))
        self.Sigma_x = numpy.zeros(shape=(self.nbStates,3,3))
        self.Wp = numpy.zeros(shape=(self.nbStates,3,3))
        self.Mu_t = numpy.linspace(0,self.nbData*self.dt,self.nbStates)
        self.Sigma_t = numpy.tile((self.nbData*self.dt/self.nbStates)*0.8,[self.nbStates,1,1])
        self.Data = numpy.zeros(shape=(9,self.nbSamples*self.nbData))
        self.loadDemonstration()
        rospy.loginfo('Loaded demonstrations')
        #                        dimensions, trajectory DoF, samples of One Demo 


    def getConfig(self):
        if rospy.has_param('learning/dmp/nbData') :
            self.nbData = rospy.get_param('learning/dmp/nbData')
        else :
            rospy.logerr('Prameter nbData not found')

        if rospy.has_param('learning/dmp/nbDataRepro') :
            self.nbDataRepro = rospy.get_param('learning/dmp/nbDataRepro')
        else :
            rospy.logerr('Prameter nbDataRepro not found')

        if rospy.has_param('learning/dmp/nbDataExport') :
            self.nbDataExport = rospy.get_param('learning/dmp/nbDataExport')
        else :
            rospy.logerr('Prameter nbDataExport not found')

        if rospy.has_param('learning/dmp/nbVar') :
            self.nbVar = rospy.get_param('learning/dmp/nbVar')
        else :
            rospy.logerr('Prameter nbVar not found')

        if rospy.has_param('learning/dmp/nbStates') :
            self.nbStates = rospy.get_param('learning/dmp/nbStates')
        else :
            rospy.logerr('Prameter nbStates not found')

        if rospy.has_param('learning/dmp/nbDataExport') :
            self.nbDataExport = rospy.get_param('learning/dmp/nbDataExport')
        else :
            rospy.logerr('Prameter nbDataExport not found')

        if rospy.has_param('learning/dmp/kV') :
            self.kV = rospy.get_param('learning/dmp/kV')
        else :
            rospy.logerr('Prameter kV not found')
            
        if rospy.has_param('learning/dmp/kP') :
            self.kP = rospy.get_param('learning/dmp/kP')
        else :
            rospy.logerr('Prameter kP not found')
                
        if rospy.has_param('learning/dmp/dt') :
            self.dt = rospy.get_param('learning/dmp/dt')
        else :
            rospy.logerr('Prameter dt not found')
        
        if rospy.has_param('learning/dmp/kPmin') :
            self.kPmin = rospy.get_param('learning/dmp/kPmin')
        else :
            rospy.logerr('Prameter kPmin not found')

        if rospy.has_param('learning/dmp/kPmax') :
            self.kPmax = rospy.get_param('learning/dmp/kPmax')
        else :
            rospy.logerr('Prameter kPmax not found')

        if rospy.has_param('learning/dmp/alpha') :
            self.alpha = rospy.get_param('learning/dmp/alpha')
        else :
            rospy.logerr('Prameter alpha not found')

        if rospy.has_param('learning/dmp/demonstration_file') :
            self.demonstration_file = rospy.get_param('learning/dmp/demonstration_file')
        else :
            rospy.logerr('Prameter demonstration_file not found')

        if rospy.has_param('learning/dmp/demonstrations') :
            self.demonstrations = rospy.get_param('learning/dmp/demonstrations')
        else :
            rospy.logerr('Prameter demonstration_file not found')

        if rospy.has_param('learning/dmp/export_filename') :
            self.exportFilename = rospy.get_param('learning/dmp/export_filename')
        else :
            rospy.logerr('Prameter export_filename not found')





    def loadDemonstration(self) :
        for n in range(self.nbSamples):
            ni=self.demonstrations[n]
            logfile = open(self.demonstration_file+"_"+str(ni)+".csv", "r").readlines()
            pose = numpy.array([[0,0,0]])
            ori = numpy.array([[0,0,0]])
            counter = 0
            for line in logfile :
                pose_aux = numpy.array([])
                ori_aux = numpy.array([])
                for word in line.split() :
                    if counter < 3 :
                        pose_aux = numpy.append(pose_aux,word)
                    else :
                        ori_aux = numpy.append(ori_aux,word)
                    counter+=1
                pose = numpy.vstack((pose,pose_aux))
                ori = numpy.vstack((ori,ori_aux))
                counter = 0
            pose = numpy.vsplit(pose,[1])[1]
            ori = numpy.vsplit(ori,[1])[1]
            nbDataTmp = pose.shape[0]-1 #shape(pose,1);
#            rospy.loginfo(nbDataTmp)
        #xx = linspace(1,nbDataTmp,nbData); 
        #d(n).Data(posEndEffectorId,:) = spline(1:nbDataTmp, posEndEffector, xx);
            xx = numpy.linspace(0, nbDataTmp, self.nbData)
#            rospy.loginfo(str(pose.T[1,:]))
#            rospy.loginfo("range "+ str(range(nbDataTmp)))
            f = interpolate.interp1d(range(nbDataTmp+1), pose.T, kind='cubic')           
            yy = f(xx)
            self.d[n,0:3,:] = yy 
            #Velocities generated from the interpolation
            #d(n).Data(velId,:) = ([d(n).Data(posId,2:end) d(n).Data(posId,end)] - d(n).Data(posId,:)) ./ m.dt;
            aux = numpy.zeros(shape=(3,self.nbData))
            aux[:,0:-1] = yy[:,1:]
            aux[:,-1]= yy[:,-1]            
            self.d[n,3:6,:] = ( ( aux - yy ) / self.dt )

            #Accelerations generated from the interpolation
            #d(n).Data(accId,:) = ([d(n).Data(velId,2:end) d(n).Data(velId,end)] - d(n).Data(velId,:)) ./ m.dt;
            aux[:,0:-1] = self.d[n,3:6,1:]
            aux[:,-1]= self.d[n,3:6,-1]            
            self.d[n,6:9,:] = ( ( aux - self.d[n,3:6,:] ) / self.dt )
            self.Data[:,((n)*self.nbData):(self.nbData*(n+1)) ] = self.d[n,:,:]
#           numpy.set_printoptions(threshold=100000)
#            rospy.loginfo('\n Values in the d data number ' + str(n) + '\n' + str(self.d[n,:,:]) + '\n' )
#            p = raw_input('wait')

            #rospy.loginfo(self.d)

    def trainningDMP(self):
        rospy.loginfo('Learning DMP ...')
        #compute weights
        s = 1; 
        #Initialization of decay term
        H = numpy.zeros(shape=(self.nbData,self.nbStates))
        h = numpy.zeros(shape=(self.nbStates))

        for n in range(self.nbData):
            s = s + (-self.alpha*s)*self.dt #Update of decay term
            t = -math.log(s)/self.alpha      
            for i in range(self.nbStates):
                h[i] = self.gaussPDF(t, self.Mu_t[i], self.Sigma_t[i,0,0]) #Probability to be in a given state                
            H[n,:] = h/numpy.sum(h) #Normalization
        #tile equivalent to repmat of matlab
        self.H = numpy.tile(H,(self.nbSamples,1)) # Repeat the process for each demonstration
        
        #Batch least norm solution to find the centers of the states (or primitives) Mu_X (Y=Mu_x*H')
        #         acc                          pos             vel
        Y = self.Data[6:9,:]*(1/self.kP) + self.Data[0:3,:] + self.Data[3:6,:]*(self.kV/self.kP)

#        rospy.loginfo(' Pseudo inversa H.t \n ' + str( numpy.linalg.pinv(self.H.T) ) + '\n' )

        self.Mu_x = numpy.dot(Y,numpy.linalg.pinv(self.H.T)) #Pseudoinverse solution Mu_x = [inv(H'*H)*H'*Y']'

        # Mu_x and H are equal to matlab. Y seem equal but Its difficult to see too many values

#------------------------ Fins Aki sembla tot okey -------------------------------------------

        #Compute residuals
        RI = numpy.eye(self.nbVar,self.nbVar)*1E-3 #Regularization term for matrix inversion
        
#        rospy.loginfo('Mu_x \n' + str(self.Mu_x) + '\n' ) 
#        casa = raw_input('Check Mu_x')
#        rospy.loginfo('H \n' + str(self.H) + '\n' )
#        casa = raw_input('Check H')
#        numpy.set_printoptions(threshold=100000)
#        rospy.loginfo('Values of Y \n' + str(Y) + '\n' )
#        casa = raw_input('Check Reshape')

        for i in range(self.nbStates) :
            a = Y-numpy.tile(self.Mu_x[:,i].reshape(len(self.Mu_x[:,i]),1),(1,self.nbData*self.nbSamples))
            b = numpy.diag(self.H[:,i])
            product = numpy.dot(a,b)
#            rospy.loginfo('Product Shape: ' + str( numpy.shape(product) ) + 'Product Shape Transpose ' + str( numpy.shape(product.T) ) + '\n' )
#            casa = raw_input('prova Reshape')
#            self.Sigma_x[:,:,i] = numpy.cov( product.reshape(len(self.Mu_x[:,i]),3) )
#            casa = raw_input('Bien')
            self.Sigma_x[i,:,:] = numpy.cov( product )
 #           rospy.loginfo('Sigma '+ str(i) + ' Value \n' + str(self.Sigma_x[:,:,i]) + '\n' )
 #           casa = raw_input('prova')
            self.Wp[i,:,:] = numpy.linalg.inv(self.Sigma_x[i,:,:]+RI) #Use variation information to determine stiffness

        # Warning sigmas are different from the matlab results I don
        

#        rospy.loginfo('Values of Sigma_x \n ' + str(self.Sigma_x) + '\n') 

        #Rescale Wp to stay within the [kPmin,kPmax] range
        V=numpy.zeros(shape=(3,3,self.nbStates))
        lambda_var = numpy.zeros(shape=(3,self.nbStates))
#        rospy.loginfo( 'Values of Wp \n' + str(self.Wp) + '\n' ) 
        for i in range(self.nbStates) :
            [Dtmp,V[i,:,:]] = numpy.linalg.eig(self.Wp[i,:,:]) #Eigencomponents decomposition
            lambda_var[:,i] = Dtmp
            # The eigen values Dtmp can be different from the matlab answers

#        rospy.loginfo( 'Values of V \n' + str(V) +'\n' ) 
        lambda_min = numpy.min(lambda_var)
        lambda_max = numpy.max(lambda_var)

        for i in range(self.nbStates) :
        #Full covariance stiffness matrix derived from residuals estimation
        #Rescale each eigenvalue such that they lie in the range [kPmin,kPmax]
            Dtmp = numpy.diag((self.kPmax-self.kPmin)*(lambda_var[:,i]-lambda_min)/(lambda_max-lambda_min) + self.kPmin);
            self.Wp[i,:,:] = numpy.dot(V[i,:,:], numpy.dot(Dtmp,numpy.linalg.inv(V[i,:,:])) ) #Reconstruction from the modified eigencomponents
        #OR
            #Standard DMP
            #self.Wp[:,:,i]=numpy.diag(numpy.ones(shape=(self.nbVar,1))*self.kP)

        rospy.loginfo('\nValues Wp \n ' + str(self.Wp) + '\n' )

        self.exportPlayData()


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


    def exportPlayData(self):
    #EXPORTPLAYDATA 
    # This function export the data to be used with the prepare ekf_Track
    # source code.
       file = open( self.exportFilename, 'w')

       file.write('kV\n' + repr(self.kV) +'\n\n')
       
       file.write('kP\n' + repr(self.kP) +'\n\n')

       file.write('Mu_t\n')
       for j in self.Mu_t:
           file.write(str(j)+' ')
       file.write('\n\n')

       file.write('Sigma_t\n')
       for i in range(self.Sigma_t.shape[0]):
           for k in range(self.Sigma_t.shape[2]):
               for j in self.Sigma_t[i,k,:]:
                   file.write(str(j)+' ')
               file.write('\n')
           file.write('\n')


 
       file.write('Mu_x\n' ) 
       
       format = numpy.zeros(shape=(numpy.shape(self.Mu_x)))

       for i in range(self.Mu_x.shape[0]):
           for j in self.Mu_x[i,:]:
               file.write(str(j)+' ')

           file.write('\n')

       file.write('\n')
       file.write('Wp\n' ) 

       format = numpy.zeros(shape=(numpy.shape(self.Wp)))

       for i in range(self.Wp.shape[0]):
           for k in range(self.Wp.shape[2]):
               for j in self.Wp[i,:,k]:
                   file.write(str(j)+' ')
               file.write('\n')
           file.write('\n')

       file.close()


if __name__ == '__main__':
    try:
        rospy.init_node('learning_dmp')
        learning_dmp = learningDmp( rospy.get_name() )
        learning_dmp.trainningDMP()
#        rospy.spin()

    except rospy.ROSInterruptException: pass