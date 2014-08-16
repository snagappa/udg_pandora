#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib
from learning_dmp_generic import LearningDmpGeneric

class HAnalyzerAndLearning:

    def __init__(self,name):
        """
        Initilize the obtject creating the node
        """
        self.name = name
        self.get_config()
        rospy.loginfo('Configuration Loaded')

    def get_config(self):
        """
        Load the configuration from the yaml file using the library
        of cola2_ros_lib
        """
        param_dict = {'samples': '/hierarchical/analyzer/samples',
                      'prefix_files': '/hierarchical/analyzer/prexif_files'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def load_all_demos(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        pass

    def find_break_points(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        pass

    def joint_sub_task(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        pass

    def learn_all_subtasks(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        # get together similar parts ???
        # com ajuntar dmp ???
        #for each sub task
        #decide with longitud and variability
         #dmp = learning_dmp_generic(lot of parameter)
         # method to check the similariy
         #while not similar change and try again
         #treball en parallel
        pass

    def run(self):
        """
        Load all the information in the different files. Compare the difference
        between them, select the time for the different sub tasks
        """
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = 'traj_auv_panel_centre'
        samples = [0,1,2,3,4,5]
        init_time = [1408001847, 1408002303, 1408002800, 1408003170, 1408003515, 1408003918]
        end_time = [1408001907, 1408002363, 1408002860, 1408003230, 1408003575, 1408003978]
        output_file_name = 'traj_auv_panel_first_aprox.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()
        # rospy.loginfo('Loading All Demos')
        # self.load_all_demos()
        # rospy.loginfo('Analizing the data')
        # self.find_break_points()
        # rospy.loginfo('Grup and select parameter for each subtask')
        # self.joint_sub_task()
        # rospy.loginfo('Start the Learning proces for each task')
        # self.learn_all_subtasks()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_hierarchical_learning", "h_analyzer_and_learning.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate h_analyzer_and_learning.yaml")

        rospy.init_node('h_analizer_and_learning')
        h_analyzer_and_learning = HAnalyzerAndLearning(rospy.get_name())
        h_analyzer_and_learning.run()
    except rospy.ROSInterruptException:
        pass
