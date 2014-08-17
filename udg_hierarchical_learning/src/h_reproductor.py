#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib

from state_machine import StateMachine
from learning_dmp_reproductor import LearningDmpReproductor

class HReproductor:

    def __init__(self, name):
        """
        Initilize the obtject creating the node
        """
        self.name = name
        self.get_config()
        #self.state_machine = self.load_state_machine(sm_file)
        rospy.loginfo('Configuration Loaded')

    def get_config(self):
        """
        Load the configuration from the yaml file using the library
        of cola2_ros_lib
        """
        param_dict = {'sm_file': '/hierarchical/reproductor/sm_file',
                      'interval_time': '/hierarchical/reproductor/interval_time'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def load_state_machine(self, file):
        """
        Load the state machine parameter
        """
        pass
        # return StateMachine()


    def run(self):
        """
        Use the state machine to reproduce the action
        """
        rate = rospy.Rate(1.0/self.interval_time)
        dmp_1 = LearningDmpReproductor(
            'Approach',
            'traj_auv_panel_first_aprox.txt',
            4,
            1.0,
            self.interval_time)
        current_pose = [0.0 , 1.0, 4.5, 0.0]
        current_vel = [0.0 , 0.0, 0.0, 0.0]
        file_sim = open("traj_simulated.csv", 'w')
        line = (repr(rospy.get_time()) + " " +
                repr(current_pose[0]) + " " +
                repr(current_pose[1]) + " " +
                repr(current_pose[2]) + " " +
                repr(current_pose[3]) + "\n")
        file_sim.write(line)
        print 'Running!!!'
        while not rospy.is_shutdown():
            [desPos, desVel] = dmp_1.generateNewPose(
                current_pose, current_vel)
            #if empty
            #print 'desPos ' + str(desPos)
            #print 'desVel ' + str(desVel)
            if len(desPos) == 0:
                break
            current_pose = desPos
            current_vel = desVel
            line = (repr(rospy.get_time()) + " " +
                    repr(current_pose[0]) + " " +
                    repr(current_pose[1]) + " " +
                    repr(current_pose[2]) + " " +
                    repr(current_pose[3]) + "\n")
            file_sim.write(line)
            rate.sleep()
        file_sim.close()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_hierarchical_learning", "h_reproductor.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate h_reproductor.yaml")

        rospy.init_node('h_reproductor')
        h_reproductor = HReproductor(rospy.get_name())
        h_reproductor.run()
    except rospy.ROSInterruptException:
        pass
