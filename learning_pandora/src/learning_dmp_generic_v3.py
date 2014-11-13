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

#use to load the external code
from learning_dmp_generic import LearningDmpGeneric

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

        # if self.Automaticks :
        #     self.kP = self.kPmin + (self.kPmax - self.kPmin)/2.0
        #     self.kV = 2.0*np.sqrt(self.kP)

        if self.Automaticks:
            self.kP = -99.0
            self.kV = -99.0

        self.kPmin = self.kPmin[0]
        self.kPmax = self.kPmax[0]


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
                      'Automaticks': 'learning/dmp/AutomaticKs',
                      'kP': 'learning/dmp/kP',
                      'kV': 'learning/dmp/kV',
                      'alpha': 'learning/dmp/alpha',
                      'demonstration_file': 'learning/dmp/demonstration_file',
                      'demonstrations': 'learning/dmp/demonstrations',
                      'export_filename': 'learning/dmp/export_filename',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.kPmin = np.asarray(self.kPmin)
        self.kPmax = np.asarray(self.kPmax)

    def run(self):
        dof_list = [1,1,1,1,0,0,0,0,0,0]
        init_time = [1415707681, 1415707973, 1415708048]
        end_time = [1415707711, 1415707999, 1415708080]
        dmp_1 = LearningDmpGeneric(self.kP, self.kV, self.kPmin, self.kPmax,
                                   self.alpha, self.nbStates, dof_list,
                                   self.nbData, self.demonstration_file,
                                   self.demonstrations,
                                   init_time, end_time, self.export_filename)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()



if __name__ == '__main__':
    try:
        rospy.init_node('learning_dmp_v2')
        learning_dmp = learningDmp(rospy.get_name())
        learning_dmp.run()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
