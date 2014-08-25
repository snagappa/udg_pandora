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
        rospy.loginfo('Learning AUV to the panel')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = 'traj_auv_panel_centre'
        samples = [0,2,3,4,5]
        init_time = [1408001847, 1408002798, 1408003174, 1408003515, 1408003918]
        end_time = [1408001877, 1408002825, 1408003217, 1408003550, 1408003967]
        output_file_name = 'traj_auv_panel_aprox.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()
        rospy.loginfo('Learned AUV to the panel')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning unfolding the arm')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = 'traj_ee_auv'
        samples = [0,2,3,4,5]
        init_time = [1408001877, 1408002825, 1408003217, 1408003550, 1408003967]
        end_time = [1408001907, 1408002857, 1408003239, 1408003603, 1408003996]
        output_file_name = 'traj_ee_auv_unfolding.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()
        rospy.loginfo('Learned unfolding the arm')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning AUV and EE grasping UVMS')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = ['traj_auv_valve_0', 'traj_auv_valve_2', 'traj_auv_valve_3',
                     'traj_auv_valve_2','traj_auv_valve_3']
        samples = [0,2,3,4,5]
        init_time = [1408001907, 1408002857, 1408003239, 1408003603, 1408003996]
        end_time = [1408001960, 1408002883, 1408003254, 1408003615, 1408004014]
        output_file_name = 'traj_auv_valve_grasping.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = ['traj_ee_valve_0', 'traj_ee_valve_2', 'traj_ee_valve_3',
                     'traj_ee_valve_2','traj_ee_valve_3']
        samples = [0,2,3,4,5]
        init_time = [1408001907, 1408002857, 1408003239, 1408003603, 1408003996]
        end_time = [1408001960, 1408002883, 1408003254, 1408003615, 1408004014]
        output_file_name = 'traj_ee_valve_grasping.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        rospy.loginfo('Learned AUV and EE grasping')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning Turning')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,1,1]
        file_name = 'force_world'
        samples = [0,2,3,4,5]
        init_time = [1408001960, 1408002883, 1408003254, 1408003615, 1408004014]
        end_time = [1408001989, 1408002938, 1408003324, 1408003653, 1408004104]
        output_file_name = 'traj_auv_panel_first_aprox.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        rospy.loginfo('Learned Turning')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning Moving AUV Out panel')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = ['traj_auv_valve_0', 'traj_auv_valve_2', 'traj_auv_valve_3',
                     'traj_auv_valve_2','traj_auv_valve_3']
        samples = [0,2,3,4,5]
        init_time = [1408001989, 1408002938, 1408003324, 1408003653, 1408004104]
        end_time = [1408001998, 1408002948, 1408003330, 1408003661, 1408004122]
        output_file_name = 'traj_auv_valve_moving_out.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        # kP = -99
        # kV = -99
        # kP_min = 1.0
        # kP_max = 4.0
        # nb_data = 500
        # alpha = 1.0
        # states = 10
        # dof_list = [1,1,1,1,0,0,0]
        # file_name = ['traj_ee_valve_0', 'traj_ee_valve_2', 'traj_ee_valve_3',
        #              'traj_ee_valve_2','traj_ee_valve_3']
        # samples = [0,1,2,3,4,5]
        # init_time = [1408001989, 1408002938, 1408003324, 1408003653, 1408004104]
        # end_time = [1408001998, 1408002948, 1408003330, 1408003661, 1408004122]
        # output_file_name = 'traj_ee_valve_moving_out.txt'

        # dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
        #                            dof_list, nb_data, file_name, samples,
        #                            init_time, end_time, output_file_name)
        # dmp_1.trainningDMP()
        # dmp_1.exportPlayData()

        rospy.loginfo('Learned moving AUV Out panel')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning Folding the Arm')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = ['traj_ee_valve_0', 'traj_ee_valve_2', 'traj_ee_valve_3',
                     'traj_ee_valve_2','traj_ee_valve_3']
        samples = [0,2,3,4,5]
        init_time = [1408001998, 1408002948, 1408003330, 1408003661, 1408004122]
        end_time = [1408002028, 1408002968, 1408003378, 1408003705, 1408004148]
        output_file_name = 'traj_ee_folding.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()

        rospy.loginfo('Learned Folding the Arm')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Learning AUV world')
        kP = -99
        kV = -99
        kP_min = 1.0
        kP_max = 4.0
        nb_data = 500
        alpha = 1.0
        states = 10
        dof_list = [1,1,1,1,0,0,0]
        file_name = 'traj_auv_panel_centre'
        samples = [0,2,3,4,5]
        init_time = [1408002028, 1408002968, 1408003378, 1408003705, 1408004148]
        end_time = [1408002100, 1408003020, 1408003402, 1408003732, 1408004202]
        output_file_name = 'traj_auv_world.txt'

        dmp_1 = LearningDmpGeneric(kP, kV, kP_min, kP_max, alpha, states,
                                   dof_list, nb_data, file_name, samples,
                                   init_time, end_time, output_file_name)
        dmp_1.trainningDMP()
        dmp_1.exportPlayData()
        rospy.loginfo('Learned AUV world')
        rospy.loginfo('****************************************************')
        rospy.loginfo('Finished!!!')


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
