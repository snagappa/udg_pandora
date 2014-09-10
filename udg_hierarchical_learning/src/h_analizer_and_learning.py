#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib
from cola2_lib import cola2_lib
from learning_dmp_generic import LearningDmpGeneric

#parse xml file
from lxml import etree

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
                      'valves': '/hierarchical/analyzer/valves',
                      'prefix_files': '/hierarchical/analyzer/prexif_files',
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def load_all_demos(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        #TODO: buscar dins a l'array les posicions bones
        self.demonstrations_auv_world = self.loadDemonstration(
            self.prefix_files[0])
        self.demonstrations_ee_auv = self.loadDemonstration(
            self.prefix_files[1])
        self.demonstrations_auv_panel_centre = self.loadDemonstration(
            self.prefix_files[2])
        self.demonstrations_force = self.loadDemonstration(
            self.prefix_files[3])
        self.demonstrations_auv_valve = []
        self.demonstrations_ee_valve = []
        for n in xrange(len(self.valves)):
            self.demonstrations_auv_valve.append(
                self.loadDemonstration(
                    self.prefix_files[5] + '_' + str(self.valves[n])))
            self.demonstrations_ee_valve.append(
                self.loadDemonstration(
                    self.prefix_files[6] + '_' + str(self.valves[n])))

    def loadDemonstration(self,file_name):
        """
        Load Demonstrations from the last point to the begining
        """
        print 'Loading Demonstrations ' + file_name + ' :'
        demonstrations = []
        for n in xrange(len(self.samples)):
            #print 'Loading Demonstration ' + file_name + "_" + str(ni)
            ni = self.samples[n]
            if type(file_name) is str:
                logfile = open(file_name + "_" + str(ni) + ".csv",
                               "r").readlines()
            else:
                #The file name is a list of elements
                logfile = open(file_name[n] + "_" + str(ni) + ".csv",
                               "r").readlines()
            # vars = np.zeros((1, self.nbVar))
            # Added the time to the var
            data_demo = np.array([[]])
            for line in logfile:
                if len(data_demo[0]) == 0:
                    data_demo = np.array([line.split()], dtype=np.float64)
                else:
                    data_demo = np.append(
                        data_demo,
                        np.array([line.split()], dtype=np.float64),
                        axis=0)
            demonstrations.append(data_demo)
        return demonstrations

    def find_break_points(self):
        """
        Load all the trajecotries, grouping them in frames and elements for each,
        sample
        """
        for n in xrange(len(self.samples)):
        #for n in xrange(1):
            print 'Analyzing Demonstration ' + str(n)
            #format time x y z roll pitch yaw
            #format2 time x y z yaw_or_ori roll pitch yaw
            #format3 time f_x f_y f_z t_x t_y t_z
            #format 1 and 2 are equal
            #TODO: generalize
            auv_world = self.demonstrations_auv_world[n]
            ee_auv = self.demonstrations_ee_auv[n]
            force = self.demonstrations_force[n]
            # Find the euclidian distance to simplify the analisis
            dist_auv_world = np.sum(np.power(auv_world[:, 1:4],2), axis=1)
            dist_ee_auv = np.sum(np.power(ee_auv[:, 1:4],2), axis=1)
            dist_force = np.sum(np.power(force[:, 1:4],2), axis=1)
            # smooth the distance
            smooth_auv_world = cola2_lib.smooth_sg(dist_auv_world, 201, 3)
            #smooth_ee_auv = cola2_lib.smooth_sg(dist_ee_auv, 51, 1)
            smooth_force = cola2_lib.smooth_sg(force[:,3], 31, 3)
            # find the break points
            [auv_max, auv_min] = cola2_lib.peakdetect(smooth_auv_world,
                                                      auv_world[:,0],
                                                      200)
            auv_max = np.array(auv_max)
            auv_min = np.array(auv_min)

            [ee_max, ee_min] = cola2_lib.peakdetect(dist_ee_auv,
                                                    ee_auv[:,0],
                                                    50)
            ee_max = np.array(ee_max)
            ee_min = np.array(ee_min)

            [force_max, force_min] = cola2_lib.peakdetect(smooth_force,
                                                          force[:,0],
                                                          150)
            force_max = np.array(force_max)
            force_min = np.array(force_min)

            ee_init, ee_end = self.find_flat_init_and_end(dist_ee_auv)

            #print 'Values ee_min ' + str(ee_min)

            if ee_max[0,1] >= dist_ee_auv[ee_init]:
                ee_min = np.append(np.array(
                    [[ee_auv[ee_init,0],dist_ee_auv[ee_init]]]),
                                   ee_min,
                                   axis = 0)
                #print 'Values ee_min ' + str(ee_min)
            else:
                ee_max = np.append(np.array(
                                        [[ee_auv[ee_init,0],
                                          dist_ee_auv[ee_init]]]),
                                   ee_max,
                                   axis = 0)

            if ee_min[-1,1] >= dist_ee_auv[ee_end]:
                ee_min = np.append(ee_min,
                                   np.array(
                                       [[ee_auv[ee_end,0],dist_ee_auv[ee_end]]]),
                                   axis = 0)
            else:
                ee_max = np.append(ee_max,
                                   np.array(
                                        [[ee_auv[ee_end,0],
                                          dist_ee_auv[ee_end]]]),
                                   axis = 0)

            import matplotlib.pyplot as plt
            f, axis = plt.subplots(3, sharex=True)
            axis[0].plot(auv_world[:,0], dist_auv_world, 'b--')
            axis[0].plot(auv_world[:,0], smooth_auv_world, 'r--')
            axis[0].plot(auv_max[:,0], auv_max[:,1], 'g*')
            axis[0].plot(auv_min[:,0], auv_min[:,1], 'g*')
            axis[1].plot(ee_auv[:,0], dist_ee_auv, 'r-')
            axis[1].plot(ee_max[:,0], ee_max[:,1], 'g*')
            axis[1].plot(ee_min[:,0], ee_min[:,1], 'g*')
            axis[2].plot(force[:,0], smooth_force,'b-')
            axis[2].plot(force_max[:,0], force_max[:,1], 'g*')
            axis[2].plot(force_min[:,0], force_min[:,1], 'g*')
            #axis[1].plot(ee_auv[ee_init,0], dist_ee_auv[ee_init], 'g+')
            #axis[1].plot(ee_auv[ee_end,0], dist_ee_auv[ee_end], 'g+')

            #plt.plot(z_axis, 'r--')
            #plt.plot(auv_world[:,0], dist_auv_world, 'b--')
            #plt.plot(auv_world[:,0], smooth_auv_world, 'r-')
            #plt.plot(auv_max[:,0], auv_max[:,1], 'g*')
            #plt.plot(auv_min[:,0], auv_min[:,1], 'g*')
            plt.show()

    def find_flat_init_and_end(self, elements):
        """
        With np.where we take the first element value and we find it over all
        the list. If there is a flat region at the beginning, or the end it
        returns the last value
        """
        #find beginning region
        first_value = elements[0]
#        same_value = np.where( np.abs(elements - first_value) < 0.1)
        same_value = np.where(np.abs(elements - first_value) < 0.001)
        #print 'Same value ' + str(same_value)
        break_point = 0
        while ( break_point < len(same_value[0])-2 and
                np.abs(same_value[0][break_point+1] -
                       same_value[0][break_point]) < 3):
            break_point += 1
        if (np.abs(same_value[0][break_point+1] -
                   same_value[0][break_point])) < 3:
            break_point_init = same_value[0][break_point+1]
        else:
            break_point_init = same_value[0][break_point]
        #find the end region
        last_value = elements[-1]
        same_value = np.where( np.abs(elements - last_value) < 0.001)
        break_point = len(same_value[0])-1
        while ( break_point > 0 and
                np.abs(same_value[0][break_point-1] -
                       same_value[0][break_point]) < 3):
            break_point -= 1
        if (np.abs(same_value[0][break_point-1] -
                   same_value[0][break_point])) < 3:
            break_point_end = same_value[0][break_point-1]
        else:
            break_point_end = same_value[0][break_point]

        return break_point_init, break_point_end


    def check_break_point(self, time, element):
        """
        Check the break point in the different data and evaluate if it is a
        change point or not. If it is store in special list and if not is
        stored in a list of degree of difficulty.
        """
        if element =='auv':
            pass
        elif element == 'ee':
            pass
        elif elemetn == 'force':
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
        output_file_name = 'turning_valve.txt'

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
        #h_analyzer_and_learning.run()
        h_analyzer_and_learning.load_all_demos()
        h_analyzer_and_learning.find_break_points()
    except rospy.ROSInterruptException:
        pass
