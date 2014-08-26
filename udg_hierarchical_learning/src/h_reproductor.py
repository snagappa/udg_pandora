#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib

#from state_machine import StateMachine
from learning_dmp_reproductor import LearningDmpReproductor

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry
#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

import threading
import tf

from lxml import etree

class HReproductor:

    def __init__(self, name):
        """
        Initilize the obtject creating the node
        """
        self.name = name
        self.get_config()
        self.load_state_machine(self.sm_file)
        #self.state_machine = self.load_state_machine(sm_file)
        rospy.loginfo('Configuration Loaded')
        # *******Vars and locks to store the data ******
        #TODO: Think to do it in exernal structure
        # Or external class can be reused for the record
        # Now is a copy paste
        self.goalPose = Pose()
        self.element_auv = Pose()
        self.element_ee = Pose()
        self.element_force = Wrench()
        self.frame_valve_0 = Pose()
        self.frame_valve_1 = Pose()
        self.frame_valve_2 = Pose()
        self.frame_valve_3 = Pose()
        self.frame_panel_centre = Pose()

        self.frame_valve_0_handle = 0.0
        self.frame_valve_1_handle = 0.0
        self.frame_valve_2_handle = 0.0
        self.frame_valve_3_handle = 0.0

        self.lock_element_auv = threading.Lock()
        self.lock_element_ee = threading.Lock()
        self.lock_element_force = threading.Lock()

        self.lock_frame_valve_0 = threading.Lock()
        self.lock_frame_valve_1 = threading.Lock()
        self.lock_frame_valve_2 = threading.Lock()
        self.lock_frame_valve_3 = threading.Lock()
        self.lock_frame_panel_centre = threading.Lock()

        self.init_element_auv = False
        self.init_element_ee = False
        self.init_element_force = False
        self.init_frame_valve_0 = False
        self.init_frame_valve_1 = False
        self.init_frame_valve_2 = False
        self.init_frame_valve_3 = False
        self.init_frame_panel_centre = False
        # ************** End of vars *****************
        # ************** Subscriber ******************
        rospy.Subscriber("/arm/pose_stamped",
                         PoseStamped,
                         self.update_element_ee,
                         queue_size = 1)

        rospy.Subscriber("/pose_ekf_slam/odometry",
                         Odometry,
                         self.update_element_auv,
                         queue_size = 1)

        rospy.Subscriber("/force_torque_controller/wrench_stamped",
                         WrenchStamped,
                         self.update_element_force,
                         queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve0",
                         PoseWithCovarianceStamped,
                         self.update_frame_valve_0,
                         queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve1",
                         PoseWithCovarianceStamped,
                         self.update_frame_valve_1,
                         queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve2",
                         PoseWithCovarianceStamped,
                         self.update_frame_valve_2,
                         queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve3",
                         PoseWithCovarianceStamped,
                         self.update_frame_valve_3,
                         queue_size = 1)

        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.update_frame_panel_centre,
                         queue_size = 1)
        # ************** End of Subscribers *************
        # ************** Publisher **********************

        #rospy.Publisher()
        # ************** End of Publishers *************

    def get_config(self):
        """
        Load the configuration from the yaml file using the library
        of cola2_ros_lib
        """
        param_dict = {'sm_file': '/hierarchical/reproductor/sm_file',
                      'interval_time': '/hierarchical/reproductor/interval_time'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

    def load_state_machine(self, file_name):
        """
        Load the state machine parameter
        """
        self.state_machine = etree.parse(file_name)
        root = self.state_machine.getroot()
        items = self.state_machine.getroot().items()
        initial_state = [s for s in items if 'initial' in s]
        if len(initial_state) == 1:
            self.initial_state = initial_state[0][1]
        else:
            rospy.logerr('Initial state not found')
            #how to shutdown
        #create an load all the states of the state machine
        list_states = self.state_machine.findall('state')
        list_of_states = []
        for state in list_states:
            #search for the invoke inside the state
            name = state.values()[0]
            element = []
            for iterator in state.iterchildren('invoke'):
                if len(iterator.getchildren()) == 3:
                    # is a DMP method
                    [frame, dof, alpha] = iterator.getchildren()
                    dmp = LearningDmpReproductor(
                        name,
                        iterator.values()[0],
                        int(dof.text),
                        float(alpha.text),
                        self.interval_time)
                    element.append(frame.text)
                    element.append(iterator.values()[1])
                    element.append(dmp)
                elif len(iterator.getchildren()) == 1:
                    element.append(iterator.getchildren()[0].text)
                    element.append('?????')
                else:
                    rospy.logerr('Not a correct format!!!!')
            conexions = []
            for iterator in state.iterchildren('transition'):
                conexions.append((iterator.keys()[0], iterator.values()[0]))
            list_of_states.append([name, element, conexions])

    # ***************** Start Subscribers ****************
    def update_frame_valve_0(self, pose_msg):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param pose_msg: Contains the position and the orientation of the vavle
        @type pose_msg: PoseWithCovarianceStamped
        """
        self.lock_frame_valve_0.acquire()
        try:
            self.frame_valve_0.position = pose_msg.pose.pose.position
            self.frame_valve_0_handle = tf.transformations.euler_from_quaternion(
                [pose_msg.pose.pose.orientation.x,
                 pose_msg.pose.pose.orientation.y,
                 pose_msg.pose.pose.orientation.z,
                 pose_msg.pose.pose.orientation.w])[2]
            if not self.init_frame_valve_0 :
                self.init_frame_valve_0 = True
        finally:
            self.lock_frame_valve_0.release()

    def update_frame_valve_1(self, pose_msg):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param pose_msg: Contains the position and the orientation of the vavle
        @type pose_msg: PoseWithCovarianceStamped
        """
        self.lock_frame_valve_1.acquire()
        try:
            self.frame_valve_1.position = pose_msg.pose.pose.position
            self.frame_valve_1_handle = tf.transformations.euler_from_quaternion(
                [pose_msg.pose.pose.orientation.x,
                 pose_msg.pose.pose.orientation.y,
                 pose_msg.pose.pose.orientation.z,
                 pose_msg.pose.pose.orientation.w])[2]
            if not self.init_frame_valve_1 :
                self.init_frame_valve_1 = True
        finally:
            self.lock_frame_valve_1.release()

    def update_frame_valve_2(self, pose_msg):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param pose_msg: Contains the position and the orientation of the vavle
        @type pose_msg: PoseWithCovarianceStamped
        """
        self.lock_frame_valve_2.acquire()
        try:
            self.frame_valve_2.position = pose_msg.pose.pose.position
            self.frame_valve_2_handle = tf.transformations.euler_from_quaternion(
                [pose_msg.pose.pose.orientation.x,
                 pose_msg.pose.pose.orientation.y,
                 pose_msg.pose.pose.orientation.z,
                 pose_msg.pose.pose.orientation.w])[2]
            if not self.init_frame_valve_2 :
                self.init_frame_valve_2 = True
        finally:
            self.lock_frame_valve_2.release()

    def update_frame_valve_3(self, pose_msg):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param pose_msg: Contains the position and the orientation of the vavle
        @type pose_msg: PoseWithCovarianceStamped
        """
        self.lock_frame_valve_3.acquire()
        try:
            self.frame_valve_3.position = pose_msg.pose.pose.position
            self.frame_valve_3_handle = tf.transformations.euler_from_quaternion(
                [pose_msg.pose.pose.orientation.x,
                 pose_msg.pose.pose.orientation.y,
                 pose_msg.pose.pose.orientation.z,
                 pose_msg.pose.pose.orientation.w])[2]
            if not self.init_frame_valve_3 :
                self.init_frame_valve_3 = True
        finally:
            self.lock_frame_valve_3.release()

    def update_frame_panel_centre(self, landMarkMap):
        """
        This method update the pose of the valve position published by the
        valve_tracker, also is update the orientation of the handle of the valve
        but is not used as the frame orientation.
        @param landMarkMap: Contains the position and the orientation of the vavle and panel
        @type landMarkMap: Map with
        """
        self.lock_frame_panel_centre.acquire()
        try:
            for mark in landMarkMap.landmark:
                if self.landmark_id == mark.landmark_id:
                    self.frame_valve_0.orientation = mark.pose.pose.orientation
                    self.frame_valve_1.orientation = mark.pose.pose.orientation
                    self.frame_valve_2.orientation = mark.pose.pose.orientation
                    self.frame_valve_3.orientation = mark.pose.pose.orientation
                    self.frame_panel_centre = mark.pose.pose
                if not self.init_frame_panel_centre :
                    self.init_frame_panel_centre = True
        finally:
            self.lock_frame_panel_centre.release()

    def update_element_auv(self, odometry):
        """
        This method update the position of the robot. Using directly the pose
        published by the pose_ekf_slam.
        @param odometry: Contains the odometry computed in the pose_ekf
        @type odometry: Odometry message
        """
        self.lock_element_auv.acquire()
        try:
            self.element_auv = odometry.pose.pose
            if not self.init_element_auv :
                self.init_element_auv = True
        finally:
            self.lock_element_auv.release()

    # def updateRollEndEffector(self, joint_state):
    #     """
    #     This method is a work around to obatin only the orientation in the roll
    #     of the end effector. This way we simply the learning because the arm for
    #     the moment can't control only the Roll in the last joint.
    #     @param joint_state: Contains an array with the position of each joint.
    #     @type joint_state: JointState message from sensor_msgs
    #     """
    #     self.lock.acquire()
    #     try:
    #         if self.initRoll :
    #             self.unnormalized_roll = self.unNormalizeAngle(
    #                 self.unnormalized_roll, joint_state.position[3])
    #         else :
    #             self.unnormalized_roll = joint_state.position[3]
    #             self.initRoll = True
    #     finally:
    #         self.lock.release()

    def update_element_ee(self, arm_pose):
        """
        This method update the pose of the end-effector in the base_arm_frame
        @param arm_pose: Contains the position and orientation of the End-effector respect the base of the arm
        @type arm_pose: PoseStamped
        """
        self.lock_element_ee.acquire()
        try:
            self.element_ee = arm_pose.pose
            if not self.init_element_ee :
                self.init_element_ee = True
        finally:
            self.lock_element_ee.release()

    def update_element_force(self, wrench_wrist):
        """
        This method update the force and torque of the end-effector,
        @param arm_pose: Contains the force and torque of the
        End-effector respect the base of the arm
        @type arm_pose: WrenchStamped
        """
        self.lock_element_force.acquire()
        try:
            self.element_force = wrench_wrist.wrench
            if not self.init_element_force :
                self.init_element_force = True
        finally:
            self.lock_element_force.release()

    def unNormalizeAngle(self, current_angle, new_angle):
        """
        This function unNormalize the Angle obtaining a continuous values
        avoiding the discontinuity, jumps from 3.14 to -3.14
        @param current_angle: contain the current angle not normalized
        @type current_angle: double
        @param new_angle: contain the new angle normalized
        @type new_angle: double
        """
        # rospy.loginfo('Current angle ' + str(current_angle) +
        #               ' New angle ' + str(new_angle))
        if abs(current_angle) > np.pi:
            #We are over one lap over
            norm_curr = cola2_lib.normalizeAngle(current_angle)
            if abs(new_angle - norm_curr) > np.pi :
                # rospy.loginfo('Overflow 2')
                if new_angle < 0.0:
                    inc0 = -1.0*(-np.pi - new_angle)
                    inc1 = -1.0*(np.pi - norm_curr)
                else:
                    inc0 = -1.0*(np.pi - new_angle)
                    inc1 = (-np.pi - norm_curr)
                return current_angle + inc0 + inc1
            else :
                # rospy.loginfo('Actual plus diff')
                return current_angle + (new_angle-norm_curr)
        else:
            if abs(new_angle - current_angle) > np.pi:
                # rospy.loginfo('Over Flow')
                if new_angle < 0.0:
                    inc0 = -1.0*(-np.pi - new_angle)
                    inc1 = -1.0*(np.pi - current_angle)
                else:
                    inc0 = -1.0*(np.pi - new_angle)
                    inc1 = (-np.pi - current_angle)
                return current_angle + inc0 + inc1
            else:
                # rospy.loginfo('Tal qual')
                return new_angle
    # ***************** End Subscribers ****************

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
        #h_reproductor.run()
    except rospy.ROSInterruptException:
        pass
