#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierachical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib

from udg_hierarchical_learning import state_machine
from udg_hierarchical_learning import learning_dmp_reproductor

class HReproductor:

    def __init__(self, name):
        """
        Initilize the obtject creating the node
        """
        self.name = name
        self.get_config()
        self.state_machine = self.load_state_machine(sm_file)
        rospy.loginfo('Configuration Loaded')

    def get_config(self):
        """
        Load the configuration from the yaml file using the library
        of cola2_ros_lib
        """
        param_dict = {'sm_file': '/hierarchical/reproductor/sm_file',
                      'interval_time': 'hierarchical/reproductor/interval_time'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def load_state_machine(self, file):
        """
        Load the state machine parameter
        """
        return StateMachine()

    def run(self):
        """
        Use the state machine to reproduce the action
        """
        rate = rospy.Rate(1.0/self.interval_time)
        while not rospy.is_shutdown():
            self.lock.release()



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
        h_reproducto = HReproductior(rospy.get_name())
        h_analyzer_and_learning.run()
    except rospy.ROSInterruptException:
        pass
