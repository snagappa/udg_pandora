#!/usr/bin/env python
"""Created on 2 December 2014
author Arnau
"""


# ROS basics
import roslib
roslib.load_manifest('udg_pandora_misc')
import rospy

# Force Torque Msgs
from geometry_msgs.msg import WrenchStamped

# Other libs
import subprocess
import threading

import numpy as np

# Params
KILLTIME = 2.0

SIZ_WINDOW = 10

class froceTorqueWatcher(object):
    '''
    Starts the guppy camera and check if messages stop to arrive.
    When that happens, kill the node and restart it.
    '''
    def __init__(self):
        '''
        Initializes logging.
        '''
        # Flag
        self.init = False
        self.called = False
        self.force_list = []
        # Subscribers
        sub_force = rospy.Subscriber(
            '/force_torque_iit/wrench_stamped', WrenchStamped, self.force_callback)
        # sub_force = rospy.Subscriber(
        #     '/force_torque_controller/wrench_stamped', WrenchStamped, self.force_callback)
        th_iit = threading.Thread(target=self.run_force_torque_iit)
        th_iit.daemon = True
        th_iit.start()
        rospy.sleep(KILLTIME)
        th_controller = threading.Thread(target=self.run_force_torque_controll)
        th_controller.daemon = True
        th_controller.start()
        self.called = True
        print('Force Torque Watcher On')

    def force_callback(self, data):
        '''
        Save data in a window of 10.
        '''
        self.force_list.append(data.wrench.torque.z)
        if len(self.force_list) > SIZ_WINDOW:
            self.force_list.pop(0)
            self.init = True

    def kill_and_reset(self):
        '''
        Kills and resets the guppy camera.
        '''
        print("Force Torque Watcher: Restarting Force Torque")

        try:
            self.called = False
            # Kill it
            ans = subprocess.call(["rosnode", "kill", "/IITForceTorque"])
            rospy.sleep(KILLTIME)
            ans = subprocess.call(["rosnode", "kill", "/force_torque_controller"])
            rospy.sleep(KILLTIME)
            # Wait for it

            # Restart node in another thread
            th_iit = threading.Thread(target=self.run_force_torque_iit)
            th_iit.daemon = True
            th_iit.start()
            rospy.sleep(KILLTIME)
            th_controller = threading.Thread(target=self.run_force_torque_controll)
            th_controller.daemon = True
            th_controller.start()
            self.called = True
        except:
            print("Problem trying to kill camera")

        # Safety
        if not self.called:
            th_iit = threading.Thread(target=self.run_force_torque_iit)
            th_iit.daemon = True
            th_iit.start()
            rospy.sleep(KILLTIME)
            th_controller = threading.Thread(target=self.run_force_torque_controll)
            th_controller.daemon = True
            th_controller.start()
            self.called = True

    def run_force_torque_iit(self):
        '''
        Just call the node.
        '''
        self.init = False
        self.force_list = []
        print("Force torque IIT: starting thread")
        subprocess.call(["roslaunch", "force_torque_iit", "forceTorque_IIT.launch"], stdin=subprocess.PIPE)
        print("Force torque IIT: Enabled")

    def run_force_torque_controll(self):
        '''
        Just call the node.
        '''
        self.init = False
        self.force_list = []
        print("Force torque Controller: starting thread")
        subprocess.call(["rosrun", "force_torque_controller", "force_torque_controller"], stdin=subprocess.PIPE)
        print("Force torque Controller: Enabled")

    def iterate(self):
        '''
        Main callback.
        '''
        # Check if initialized
        if (self.init and
            np.all(np.asarray(self.force_list)==self.force_list[0])):
                print("GuppyKiller: Too much time to init...")
                self.kill_and_reset()


if __name__ == '__main__':
    # Run the node
    rospy.init_node('force_torque_watcher')
    watcher = froceTorqueWatcher()
    # Spin until ready to move
    while not rospy.is_shutdown():
        rospy.sleep(1.0)
        watcher.iterate()
