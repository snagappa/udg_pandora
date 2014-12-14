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
            self.kP = np.ones(len(self.kP))*-99.0
            self.kV = np.ones(len(self.kV))*-99.0

        # self.kPmin_auv_x = self.kPmin[0]
        # self.kPmax_auv_x = self.kPmax[0]

        # self.kPmin_auv_z = self.kPmin[1]
        # self.kPmax_auv_z = self.kPmax[1]

        # self.kPmin_ee_x = self.kPmin[2]
        # self.kPmax_ee_x = self.kPmax[2]

        # self.kPmin_ee_z = self.kPmin[3]
        # self.kPmax_ee_z = self.kPmax[3]

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
        self.kPmin_auv_x = np.asarray(self.kPmin)
        self.kPmax_auv_x = np.asarray(self.kPmax)

    def run(self):
        # v2
        # init_time = [1415707681, 1415707973, 1415708048]
        # end_time = [1415707711, 1415707999, 1415708080]
        # v3 0-2
        # init_time = [1416044166, 1416044428, 1416045022]
        # end_time = [1416044321, 1416044560, 1416045136]

        # v3 3-6 - long
        # init_time = [1416135344, 1416135577, 1416135787, 1416136091]
        # end_time = [1416135477, 1416135699, 1416135912, 1416136210]

        # v3 7-8 shor
        # init_time = [1416214260, 1416214715]
        # end_time = [1416214380, 1416214812]

        # v3 Sim 0-1
        # init_time = [141647592, 1416476142]
        # end_time = [1416475994, 1416476201]

        # v3 Real 10-11
        # init_time = [1416493453, 1416493585]
        # end_time = [1416493669, 1416493774]

        # v3 Short 15-17
        # init_time = [1416562745, 1416562865, 1416563101]
        # end_time = [1416562818, 1416562947, 1416563160]

        # v3 Short 16-17
        # init_time = [1416562865, 1416563101]
        # end_time = [1416562947, 1416563160]

        # v3 Short 19 21
        # init_time = [1416820392, 1416905126]
        # end_time = [1416820489, 1416905203]

        # v3 Short 30 31
        # init_time = [1416933728, 1416933910]
        # end_time = [1416933822, 1416933997]

        # v3 Short Correct ori 35 36
        # init_time = [1416990052, 1416990248]
        # end_time = [1416990120, 1416990325]

        # v3 Short Correct ori 37 38 39 40
        # init_time = [1416999900, 1417000037,1417000224, 1417000336]
        # end_time = [1416999987, 1417000134, 1417000286, 1417000419]

        # v3 Short Correct ori and yaw robot 45,46,47
        # init_time = [1417078150, 1417078307, 1417078423]
        # end_time = [1417078235, 1417078384, 1417078488]

        # v3 Short Correct ori and yaw robot 46,47
        # init_time = [1417078307, 1417078423]
        # end_time = [1417078374, 1417078488]

        # v3 Short Correct ori and yaw robot 50,51
        # init_time = [1417700808, 1417700942]
        # end_time = [1417700873, 1417701015]

        # v3 Short Correct ori and yaw robot 60,61,62
        # init_time = [1418381955, 1418382242, 1418382383]
        # end_time = [1418382040, 1418382304, 1418382446]

        # v3 Short Correct ori and yaw robot 61,62
        # init_time = [1418382242, 1418382383]
        # end_time = [1418382304, 1418382446]

        # v3 Short Correct ori and yaw robot 70, 71, 72
        init_time = [1418556105, 1418556507, 1418556965]
        end_time = [1418556193, 1418556594, 1418557044]

        # AUV X Y Yaw
        dof_list = [1,1,0,1,0,0,0,0,0,0]
        dmp_1 = LearningDmpGeneric(self.kP[0], self.kV[0], self.kPmin[0], self.kPmax[0],
                                   self.alpha, self.nbStates[0], dof_list,
                                   self.nbData, self.demonstration_file,
                                   self.demonstrations,
                                   init_time, end_time, self.export_filename[0])
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        # AUV Z
        dof_list = [0,0,1,0,0,0,0,0,0,0]
        dmp_1 = LearningDmpGeneric(self.kP[1], self.kV[1], self.kPmin[1], self.kPmax[1],
                                   self.alpha, self.nbStates[1], dof_list,
                                   self.nbData, self.demonstration_file,
                                   self.demonstrations,
                                   init_time, end_time, self.export_filename[1])
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        # EE X and Y
        #dof_list = [0,0,0,0,1,1,0,1,0,0]
        dof_list = [0,0,0,0,1,1,0,0,0,1]
        dmp_1 = LearningDmpGeneric(self.kP[2], self.kV[2], self.kPmin[2], self.kPmax[2],
                                   self.alpha, self.nbStates[2], dof_list,
                                   self.nbData, self.demonstration_file,
                                   self.demonstrations,
                                   init_time, end_time, self.export_filename[2])
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        # EE Z
        dof_list = [0,0,0,0,0,0,1,0,0,0]
        dmp_1 = LearningDmpGeneric(self.kP[3], self.kV[3], self.kPmin[3], self.kPmax[3],
                                   self.alpha, self.nbStates[3], dof_list,
                                   self.nbData, self.demonstration_file,
                                   self.demonstrations,
                                   init_time, end_time, self.export_filename[3])
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
