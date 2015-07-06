#!/usr/bin/env python

#ROS imports
import roslib
roslib.load_manifest('udg_parametric_learning')
import rospy

#use to load the configuration function
from cola2_lib import cola2_ros_lib

import numpy as np
import math

#from learning_dmp_parametric import LearningDmpParametric
from learning_dmp_generic_v2 import LearningDmpGeneric

class LearningDmpParametricAuto(object):
    """
    This class execute an exploration parameters where the Kp KV and
    number of states are find with exploration method.
    Future work will include the search using PoWER
    """
    #def __init__(self, kP_min, kP_max, alpha, states_min, states_max, )
    def __init__(self, name):
        """
        Iinitialize the class
        """
        self.get_config()
        self.dmp_parametric = LearningDmpParametric()

    def get_config(self):
        """
        This function loads all the parameter form the rosparam server using the
        function developed in the cola2_ros_lib.
        """
        param_dict = {'kp_min': 'learning/auto/kp_min',
                      'kp_max': 'learning/auto/kp_max',
                      'state_min': 'learning/auto/state_min',
                      'state_max': 'learning/auto/state_max',
                      'step_kp': 'learning/auto/step_kp',
                      'step_state': 'learning/auto/step_state'
                      'alpha': 'learning/auto/alpha',
                      'nbVar': 'learning/auto/nbVar',
                      'demonstration_file': 'learning/auto/demonstration_file',
                      'demonstrations': 'learning/auto/demonstrations',
                      'param_value': 'learning/auto/param_value',
                      'param_samples': 'learning/auto/param_samples',
                      'init_time': 'learning/auto/init_time',
                      'end_time': 'learning/auto/end_time',
                      'interval_time': 'learning/auto/interval_time',
                      'learning_export_file': 'learning/auto/learning_export_file',
                      'nbDataRepro': 'learning/reproductor/complete/nbDataRepro',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def reproduce_trajectory(self):
        pass

    def compare_trajectory(self):
        pass

    def learn_and_reproduce(self, kp_min, kp_max, states):
        '''
        The parameters are learned using the Dmp parameters and not exported
        '''
        dof_list = [1,0,0,1,0,0,0,0,0,0]
        dmp_auv_x_yaw = LearningDmpGeneric(-99.0, -99.0, kp_min, kp_max,
                                           self.alpha, states, dof_list,
                                           nbData, self.demonstration_file,
                                           self.demonstrations,
                                           self.init_time, self.end_time,
                                           self.learning_export_file,
                                           self.interval_time)
        dof_list = [0,1,0,0,0,0,0,0,0,0]
        dmp_auv_y = LearningDmpGeneric(-99.0, -99.0, kp_min, kp_max,
                                       self.alpha, states, dof_list,
                                       nbData, self.demonstration_file,
                                       self.demonstrations,
                                       self.init_time, self.end_time,
                                       self.learning_export_file,
                                       self.interval_time)
        dof_list = [0,0,1,0,0,0,0,0,0,0]
        dmp_auv_z = LearningDmpGeneric(-99.0, -99.0, kp_min, kp_max,
                                       self.alpha, states, dof_list,
                                       nbData, self.demonstration_file,
                                       self.demonstrations,
                                       self.init_time, self.end_time,
                                       self.learning_export_file,
                                       self.interval_time)
        dof_list = [0,0,0,0,1,1,0,0,0,1]
        dmp_ee_x_y_yaw = LearningDmpGeneric(-99.0, -99.0, kp_min, kp_max,
                                            self.alpha, states, dof_list,
                                            nbData, self.demonstration_file,
                                            self.demonstrations,
                                            self.init_time, self.end_time,
                                            self.learning_export_file,
                                            self.interval_time)
        dof_list = [0,0,0,0,0,0,1,0,0,0]
        dmp_ee_z = LearningDmpGeneric(-99.0, -99.0, kp_min, kp_max,
                                      self.alpha, states, dof_list,
                                      nbData, self.demonstration_file,
                                      self.demonstrations,
                                      self.init_time, self.end_time,
                                      self.learning_export_file,
                                      self.interval_time)

    def searching_parameter(self):
        '''
        Searching for the parameter
        '''
        kp_list = np.arange(self.kp_min, self.kp_max, self.step_kp)
        state_list = np.arange(self.state_min, self.state_max, self.step_state)
        # for the moment we focus on the Kp list
        for kp_min in kp_list:
            Kp_min_index = np.where(kp_list==kp_min)[0][0] + 1
            if kp_min_index == np.size(kp_min_index):
                break
            for kp_max in Kp_list[kp_min_index:-1]:
                for n in states:
                    self.learn_and_reproduce(kp_min, kp_max, n)

    def __del__(self):
        print 'Parameters Adjusted'

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_parametric_learning", "learning_reproductor_action_v2.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate learning_reproductor_action.yaml")

        rospy.init_node('learning_reproductor_action')
        learning_reproductor = learningReproductorAct(rospy.get_name())
        learning_reproductor.play()
#        rospy.spin()
    except rospy.ROSInterruptException:
        pass
