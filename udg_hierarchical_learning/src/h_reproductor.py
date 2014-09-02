#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib
from cola2_lib import cola2_lib
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

        #TODO: fer-ho amb un vector ???
        self.lock_frame_valve_0 = threading.Lock()
        self.lock_frame_valve_1 = threading.Lock()
        self.lock_frame_valve_2 = threading.Lock()
        self.lock_frame_valve_3 = threading.Lock()
        self.lock_frame_panel_centre = threading.Lock()

        if not self.simulation:
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

            #TODO: fer-ho amb un for fer la funcio generica pas de parametre
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
        else:
            # ************** AUV **********************
            self.element_auv.position.x = 0.0
            self.element_auv.position.y = 0.0
            self.element_auv.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.element_auv.orientation.x = quaternion[0]
            self.element_auv.orientation.y = quaternion[1]
            self.element_auv.orientation.z = quaternion[2]
            self.element_auv.orientation.w = quaternion[3]

            # ************** EE **********************
            self.element_ee.position.x = 0.0
            self.element_ee.position.y = 0.0
            self.element_ee.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.element_ee.orientation.x = quaternion[0]
            self.element_ee.orientation.y = quaternion[1]
            self.element_ee.orientation.z = quaternion[2]
            self.element_ee.orientation.w = quaternion[3]

            # ************** Force **********************
            self.element_force.force.x = 0.0
            self.element_force.force.y = 0.0
            self.element_force.force.z = 0.0
            self.element_force.torque.x = 0.0
            self.element_force.torque.y = 0.0
            self.element_force.torque.z = 0.0

            # ************** Valve_0 **********************
            self.frame_valve_0.position.x = 0.0
            self.frame_valve_0.position.y = 0.0
            self.frame_valve_0.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.frame_valve_0.orientation.x = quaternion[0]
            self.frame_valve_0.orientation.y = quaternion[1]
            self.frame_valve_0.orientation.z = quaternion[2]
            self.frame_valve_0.orientation.w = quaternion[3]
            self.frame_valve_0_handle = 0.0

            # ************** Valve_1 **********************
            self.frame_valve_1.position.x = 0.0
            self.frame_valve_1.position.y = 0.0
            self.frame_valve_1.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.frame_valve_1.orientation.x = quaternion[0]
            self.frame_valve_1.orientation.y = quaternion[1]
            self.frame_valve_1.orientation.z = quaternion[2]
            self.frame_valve_1.orientation.w = quaternion[3]
            self.frame_valve_1_handle = 0.0

            # ************** Valve_2 **********************
            self.frame_valve_2.position.x = 0.0
            self.frame_valve_2.position.y = 0.0
            self.frame_valve_2.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.frame_valve_2.orientation.x = quaternion[0]
            self.frame_valve_2.orientation.y = quaternion[1]
            self.frame_valve_2.orientation.z = quaternion[2]
            self.frame_valve_2.orientation.w = quaternion[3]
            self.frame_valve_2_handle = 0.0

            # ************** Valve_3 **********************
            self.frame_valve_3.position.x = 0.0
            self.frame_valve_3.position.y = 0.0
            self.frame_valve_3.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.frame_valve_3.orientation.x = quaternion[0]
            self.frame_valve_3.orientation.y = quaternion[1]
            self.frame_valve_3.orientation.z = quaternion[2]
            self.frame_valve_3.orientation.w = quaternion[3]
            self.frame_valve_3_handle = 0.0

            # ************** panel_centre **********************
            self.frame_panel_centre.position.x = 0.0
            self.frame_panel_centre.position.y = 0.0
            self.frame_panel_centre.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(
                0.0, 0.0, 0.0)
            self.frame_panel_centre.orientation.x = quaternion[0]
            self.frame_panel_centre.orientation.y = quaternion[1]
            self.frame_panel_centre.orientation.z = quaternion[2]
            self.frame_panel_centre.orientation.w = quaternion[3]

            self.init_element_auv = True
            self.init_element_ee = True
            self.init_element_force = True
            self.init_frame_valve_0 = True
            self.init_frame_valve_1 = True
            self.init_frame_valve_2 = True
            self.init_frame_valve_3 = True
            self.init_frame_panel_centre = True

            #Files to store the trajectory
            self.file_ee_valve_0 = open("sim_traj_ee_valve_0.csv", 'w')
            self.file_ee_valve_1 = open("sim_traj_ee_valve_1.csv", 'w')
            self.file_ee_valve_2 = open("sim_traj_ee_valve_2.csv", 'w')
            self.file_ee_valve_3 = open("sim_traj_ee_valve_3.csv", 'w')
            self.file_ee_panel_centre = open("sim_traj_ee_panel_centre.csv",
                                             'w')
            self.file_ee_auv = open("sim_traj_ee_auv.csv", 'w')
            self.file_ee_world = open("sim_traj_ee_world.csv", 'w')

            self.file_auv_valve_0 = open("sim_traj_auv_valve_0.csv", 'w')
            self.file_auv_valve_1 = open("sim_traj_auv_valve_1.csv", 'w')
            self.file_auv_valve_2 = open("sim_traj_auv_valve_2.csv", 'w')
            self.file_auv_valve_3 = open("sim_traj_auv_valve_3.csv", 'w')
            self.file_auv_panel_centre = open("sim_traj_auv_panel_centre.csv",
                                             'w')
            self.file_auv_world = open("sim_traj_auv_world.csv", 'w')

            self.file_force_ee = open("sim_force_ee.csv", 'w')

    def get_config(self):
        """
        Load the configuration from the yaml file using the library
        of cola2_ros_lib
        """
        param_dict = {'sm_file': '/hierarchical/reproductor/sm_file',
                      'interval_time': '/hierarchical/reproductor/interval_time',
                      'simulation': '/hierarchical/reproductor/simulation',
                      'enabled': '/hierarchical/reproductor/enabled',
                      'goal_valve': '/hierarchical/reproductor/goal_valve',
                      'base_pose': '/arm_controller/base_pose',
                      'base_ori': '/arm_controller/base_ori'
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
        self.list_of_states = []
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
                conexions.append([iterator.values()[0], iterator.values()[1]])
            self.list_of_states.append([name, element, conexions])

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

    def obtain_current_pose(self, frame, dmp_type, prev_pose):
        """
        Computes the position of the element related with the desired frame
        @param frame: name of the frame
        @type frame: string
        @param dmp_type: name of the dmp type
        @type dmp_type: string
        @param prev_pose: previous position
        @type prev_pose: numpy array
        """
        current_pose = np.array([])
        if dmp_type == 'dmp_ee':
            current_pose = self.compute_pose_ee(frame)
        elif dmp_type == 'dmp_auv':
            current_pose = self.compute_pose_auv(frame)
        elif dmp_type == 'dmp_force':
            current_pose = self.compute_forec(frame)
        else:
            rospy.logerr('Not a valid DMP type ' + str(dmp_type))
        return current_pose

    def compute_pose_ee(self, frame):
        """
        Compute the position of the end-effector in the frame
        @param frame: name of the frame
        @type frame: string
        """
        if self.init_element_ee and self.init_element_auv:
            self.lock_element_ee.acquire()
            self.lock_element_auv.acquire()
            try:
                base_pose = Pose()
                quaterninon = tf.transformations.quaternion_from_euler(
                    self.base_ori[0],
                    self.base_ori[1],
                    self.base_ori[2])

                base_pose.orientation.x = quaterninon[0]
                base_pose.orientation.y = quaterninon[1]
                base_pose.orientation.z = quaterninon[2]
                base_pose.orientation.w = quaterninon[3]

                base_pose.position.x = self.base_pose[0]
                base_pose.position.y = self.base_pose[1]
                base_pose.position.z = self.base_pose[2]

                arm_base = self.convert_pose_same_target_origin(
                    self.element_ee, base_pose)
                if frame == 'auv':
                    euler = tf.transformations.euler_from_quaternion([
                        arm_base.orientation.x,
                        arm_base.orientation.y,
                        arm_base.orientation.z,
                        arm_base.orientation.w])
                    return np.array([
                        arm_base.position.x,
                        arm_base.position.y,
                        arm_base.position.z,
                        euler[0],
                        euler[1],
                        euler[2]])
                arm_world = self.convert_pose_same_target_orgin(
                    arm_base, self.element_auv)
                if frame == 'world':
                    euler = tf.transformations.euler_from_quaternion([
                        arm_world.orientation.x,
                        arm_world.orientation.y,
                        arm_world.orientation.z,
                        arm_world.orientation.w])
                    return np.array([
                        arm_world.position.x,
                        arm_world.position.y,
                        arm_world.position.z,
                        euler[0],
                        euler[1],
                        euler[2]])
                rot_work_around = tf.transformations.euler_matrix(
                    0,np.pi/2.0,-np.pi/2.0)
                ori_panel = tf.transformations.quaternion_matrix(
                    [self.frame_panel_centre.orientation.x,
                     self.frame_panel_centre.orientation.y,
                     self.frame_panel_centre.orientation.z,
                     self.frame_panel_centre.orientation.w]
                )
                new_panel = np.dot(ori_panel, rot_work_around)
                ori_work_around = tf.transformations.euler_from_matrix(
                    new_panel)[2]
            finally:
                self.lock_element_auv.release()
                self.lock_element_ee.release()
            if frame == 'panel_centre':
                if self.init_frame_panel_centre:
                    self.lock_frame_panel_centre.acquire()
                    try:
                        [arm_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            arm_world,
                            self.frame_panel_centre,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            arm_world.orientation.x,
                            arm_world.orientation.y,
                            arm_world.orientation.z,
                            arm_world.orientation.w])
                        return np.array([
                            arm_world.position.x,
                            arm_world.position.y,
                            arm_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_panel_centre.release()
                else:
                    rospy.logerr('Panel centre is not initialized')
                    return []
            if frame == 'valve' and self.goal_valve == 0:
                if self.init_frame_valve_0:
                    self.lock_frame_valve_0.acquire()
                    try:
                        [arm_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            arm_world,
                            self.frame_valve_0,
                            self.frame_valve_0_handle)
                        euler = tf.transformations.euler_from_quaternion([
                            arm_world.orientation.x,
                            arm_world.orientation.y,
                            arm_world.orientation.z,
                            arm_world.orientation.w])
                        return np.array([
                            arm_world.position.x,
                            arm_world.position.y,
                            arm_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_1.release()
                else:
                    rospy.logerr('Valve 0 is not initialized')
                    return []
            if frame == 'valve' and self.goal_valve == 1:
                if self.init_frame_valve_1:
                    self.lock_frame_valve_1.acquire()
                    try:
                        [arm_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            arm_world,
                            self.frame_valve_1,
                            self.frame_valve_1_handle)
                        euler = tf.transformations.euler_from_quaternion([
                            arm_world.orientation.x,
                            arm_world.orientation.y,
                            arm_world.orientation.z,
                            arm_world.orientation.w])
                        return np.array([
                            arm_world.position.x,
                            arm_world.position.y,
                            arm_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_1.release()
                else:
                    rospy.logerr('Valve 1 is not initialized')
                    return []
            elif frame == 'valve' and self.goal_valve == 2:
                if self.init_frame_valve_2:
                    self.lock_frame_valve_2.acquire()
                    try:
                        [arm_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            arm_world,
                            self.frame_valve_2,
                            self.frame_valve_2_handle)
                        euler = tf.transformations.euler_from_quaternion([
                            arm_world.orientation.x,
                            arm_world.orientation.y,
                            arm_world.orientation.z,
                            arm_world.orientation.w])
                        return np.array([
                            arm_world.position.x,
                            arm_world.position.y,
                            arm_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_2.release()
                else:
                    rospy.logerr('Valve 3 is not initialized')
                    return []
            elif frame == 'valve' and self.goal_valve == 3:
                if self.init_frame_valve_3:
                    self.lock_frame_valve_3.acquire()
                    try:
                        [arm_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            arm_world,
                            self.frame_valve_3,
                            self.frame_valve_3_handle)
                        euler = tf.transformations.euler_from_quaternion([
                            arm_world.orientation.x,
                            arm_world.orientation.y,
                            arm_world.orientation.z,
                            arm_world.orientation.w])
                        return np.array([
                            arm_world.position.x,
                            arm_world.position.y,
                            arm_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_3.release()
                else:
                    rospy.logerr('Valve 3 is not initialized')
                    return []
            else:
                rospy.logerr('Not a valid Frame defined')
                return []
        else:
            rospy.logerr('AUV or EE are not initialized')
            return []

    def compute_pose_auv(self, frame):
        """
        Compute the position of the end-effector in the frame
        @param frame: name of the frame
        @type frame: string
        """
        if self.init_element_auv:
            self.lock_element_auv.acquire()
            try:
                if frame == 'world':
                    euler = tf.transformations.euler_from_quaternion([
                        self.element_auv.orientation.x,
                        self.element_auv.orientation.y,
                        self.element_auv.orientation.z,
                        self.element_auv.orientation.w])
                    return np.array([
                        self.element_auv.position.x,
                        self.element_auv.position.y,
                        self.element_auv.position.z,
                        euler[0],
                        euler[1],
                        euler[2]])
                rot_work_around = tf.transformations.euler_matrix(
                    0,np.pi/2.0,-np.pi/2.0)
                self.lock_frame_panel_centre.acquire()
                try:
                    ori_panel = tf.transformations.quaternion_matrix(
                        [self.frame_panel_centre.orientation.x,
                         self.frame_panel_centre.orientation.y,
                         self.frame_panel_centre.orientation.z,
                         self.frame_panel_centre.orientation.w]
                    )
                finally:
                    self.lock_frame_panel_centre.release()
                new_panel = np.dot(ori_panel, rot_work_around)
                ori_work_around = tf.transformations.euler_from_matrix(
                    new_panel)[2]
                auv_world = self.element_auv
            finally:
                self.lock_element_auv.release()
            if frame == 'panel_centre':
                if self.init_frame_panel_centre:
                    self.lock_frame_panel_centre.acquire()
                    try:
                        [auv_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            auv_world,
                            self.frame_panel_centre,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            auv_world.orientation.x,
                            auv_world.orientation.y,
                            auv_world.orientation.z,
                            auv_world.orientation.w])
                        return np.array([
                            auv_world.position.x,
                            auv_world.position.y,
                            auv_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_panel_centre.release()
                else:
                    rospy.logerr('Panel centre is not initialized')
                    return []
            if frame == 'valve' and self.goal_valve == 0:
                if self.init_frame_valve_0:
                    self.lock_frame_valve_0.acquire()
                    try:
                        [auv_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            auv_world,
                            self.frame_valve_0,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            auv_world.orientation.x,
                            auv_world.orientation.y,
                            auv_world.orientation.z,
                            auv_world.orientation.w])
                        return np.array([
                            auv_world.position.x,
                            auv_world.position.y,
                            auv_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_0.release()
                else:
                    rospy.logerr('Valve 0 is not initialized')
                    return []
            elif frame == 'valve' and self.goal_valve == 1:
                if self.init_frame_valve_1:
                    self.lock_frame_valve_1.acquire()
                    try:
                        [auv_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            auv_world,
                            self.frame_valve_1,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            auv_world.orientation.x,
                            auv_world.orientation.y,
                            auv_world.orientation.z,
                            auv_world.orientation.w])
                        return np.array([
                            auv_world.position.x,
                            auv_world.position.y,
                            auv_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_1.release()
                else:
                    rospy.logerr('Valve 1 is not initialized')
                    return []
            elif frame == 'valve' and self.goal_valve == 2:
                if self.init_frame_valve_2:
                    self.lock_frame_valve_2.acquire()
                    try:
                        [auv_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            auv_world,
                            self.frame_valve_2,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            auv_world.orientation.x,
                            auv_world.orientation.y,
                            auv_world.orientation.z,
                            auv_world.orientation.w])
                        return np.array([
                            auv_world.position.x,
                            auv_world.position.y,
                            auv_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_2.release()
                else:
                    rospy.logerr('Valve 3 is not initialized')
                    return []
            elif frame == 'valve' and self.goal_valve == 3:
                if self.init_frame_valve_3:
                    self.lock_frame_valve_3.acquire()
                    try:
                        [auv_panel, ori_ee_panel] = self.convert_pose_same_orgin(
                            auv_world,
                            self.frame_valve_3,
                            ori_work_around)
                        euler = tf.transformations.euler_from_quaternion([
                            auv_world.orientation.x,
                            auv_world.orientation.y,
                            auv_world.orientation.z,
                            auv_world.orientation.w])
                        return np.array([
                            auv_world.position.x,
                            auv_world.position.y,
                            auv_world.position.z,
                            ori_ee_panel,
                            euler[0],
                            euler[1],
                            euler[2]])
                    finally:
                        self.lock_frame_valve_3.release()
                else:
                    rospy.logerr('Valve 3 is not initialized')
                    return []
            else:
                rospy.logerr('Not a valid Frame defined')
                return []
        else:
            rospy.logerr('AUV or EE are not initialized')
            return []

    def compute_force(self, frame):
        """
        Compute the position of the end-effector in the frame
        @param frame: name of the frame
        @type frame: string
        """
        if self.init_element_force:
            self.lock_element_force.acquire()
            try:
                if frame == 'end_effector':
                    return np.array([
                        self.element_force.force.x,
                        self.element_force.force.y,
                        self.element_force.force.z,
                        self.element_force.torque.x,
                        self.element_force.torque.y,
                        self.element_force.torque.z
                    ])
                else:
                    rospy.logerr('Frame not correct for the force')
            finally:
                self.lock_element_force.release()
        else:
            rospy.logerr('Force not initialized')

    def convert_pose_same_target_origin(self, pose, new_origin):
        """
        This function convert the first pose and orientation in the world frame,
        to the frame passed as second argument. Finally it store it at the file
        passed as third parameter.
        @param pose: Position and Orientation of the element in world
        @type pose: Pose
        @param frame: Position and Orientation of the frame
        @type frame: Pose
        @return ret: Return the new position written in the file
        @tyep ret: Pose
        """
        pose_frame = tf.transformations.quaternion_matrix(
            [pose.orientation.x,
             pose.orientation.y,
             pose.orientation.z,
             pose.orientation.w]
        )

        pose_frame[0, 3] = pose.position.x
        pose_frame[1, 3] = pose.position.y
        pose_frame[2, 3] = pose.position.z

        frame = tf.transformations.quaternion_matrix(
            [new_origin.orientation.x,
             new_origin.orientation.y,
             new_origin.orientation.z,
             new_origin.orientation.w]
        )

        frame[0, 3] = new_origin.position.x
        frame[1, 3] = new_origin.position.y
        frame[2, 3] = new_origin.position.z

        new_frame = np.dot(frame, pose_frame)

        new_pose = Pose()
        quaternion = tf.transformations.quaternion_from_matrix(new_frame)
        new_pose.orientation.x = quaternion[0]
        new_pose.orientation.y = quaternion[1]
        new_pose.orientation.z = quaternion[2]
        new_pose.orientation.w = quaternion[3]

        new_pose.position.x = new_frame[0, 3]
        new_pose.position.y = new_frame[1, 3]
        new_pose.position.z = new_frame[2, 3]

        return new_pose

    def convert_pose_same_orgin(self, element, frame, ori):
        """
        This function convert the first pose and orientation in the world frame,
        to the frame passed as second argument. Includes new extra orientation
        to simplify the learning process Finally it store it at the file
        passed as third parameter.
        @param pose: Position and Orientation of the element in world
        @type pose: Pose
        @param frame: Position and Orientation of the frame
        @type frame: Pose
        @param file: Position and Orientation of the element
        @type ori: Orientation in yaw of the valve or the panel
        @param ori: Double
        @return ret: Return the new position wroted in the file
        @tyep ret: Pose
        """
        element_frame = tf.transformations.quaternion_matrix(
            [element.orientation.x,
             element.orientation.y,
             element.orientation.z,
             element.orientation.w])

        element_frame[0, 3] = element.position.x
        element_frame[1, 3] = element.position.y
        element_frame[2, 3] = element.position.z

        trans_matrix = tf.transformations.quaternion_matrix(
            [frame.orientation.x,
             frame.orientation.y,
             frame.orientation.z,
             frame.orientation.w])

        trans_matrix[0, 3] = frame.position.x
        trans_matrix[1, 3] = frame.position.y
        trans_matrix[2, 3] = frame.position.z

        #invert Matrix
        inv_mat = np.zeros([4, 4])
        inv_mat[3, 3] = 1.0
        inv_mat[0:3, 0:3] = np.transpose(trans_matrix[0:3, 0:3])
        inv_mat[0:3, 3] = np.dot((-1*inv_mat[0:3, 0:3]),
                                 trans_matrix[0:3, 3])

        new_frame = np.dot(inv_mat, element_frame)

        yaw_element = tf.transformations.euler_from_matrix(element_frame)[2]
        difference = cola2_lib.normalizeAngle(yaw_element - ori)

        new_pose = Pose()
        quaternion = tf.transformations.quaternion_from_matrix(new_frame)
        new_pose.orientation.x = quaternion[0]
        new_pose.orientation.y = quaternion[1]
        new_pose.orientation.z = quaternion[2]
        new_pose.orientation.w = quaternion[3]

        new_pose.position.x = new_frame[0, 3]
        new_pose.position.y = new_frame[1, 3]
        new_pose.position.z = new_frame[2, 3]

        euler = tf.transformations.euler_from_matrix(new_frame)

        return [new_pose, difference]

    def store_pose(self, current_pose, frame, element):
        """
        Record the trajectory of the different elements.
        @param current_pose: Actual position in the desired frame
        @type current_pose: array different sizes
        @param frame: Name of the frame
        @type frame: String
        @param element: Element recorded (ee, auv or force)
        @type element: String
        """
        line = repr(rospy.get_time()) + ' '
        for pose in current_pose:
            line += repr(pose) + ' '
        line += '\n'
        if frame == 'valve':
            if self.goal_valve == 0:
                if element.find('ee') != -1:
                    self.file_ee_valve_0.write(line)
                elif element.find('auv') != -1:
                    self.file_auv_valve_0.write(line)
                else:
                    rospy.logerr('Error the element is not valid')
            elif self.goal_valve == 1:
                if element.find('ee') != -1:
                    self.file_ee_valve_1.write(line)
                elif element.find('auv') != -1:
                    self.file_auv_valve_1.write(line)
                else:
                    rospy.logerr('Error the element is not valid')
            elif self.goal_valve == 2:
                if element.find('ee') != -1:
                    self.file_ee_valve_2.write(line)
                elif element.find('auv') != -1:
                    self.file_auv_valve_2.write(line)
                else:
                    rospy.logerr('Error the element is not valid')
            elif self.goal_valve == 3:
                if element.find('ee') != -1:
                    self.file_ee_valve_3.write(line)
                elif element.find('auv') != -1:
                    self.file_auv_valve_3.write(line)
                else:
                    rospy.logerr('Error the element is not valid')
            else:
                rospy.logerr('Error the goal is not valid')
        elif frame == 'panel_centre':
            if element.find('ee') != -1:
                self.file_ee_panel_centre.write(line)
            elif element.find('auv') != -1:
                self.file_auv_panel_centre.write(line)
            else:
                rospy.logerr('Error the element is not valid')
        elif frame == 'auv':
            if element.find('ee') != -1:
                self.file_ee_auv.write(line)
            else:
                rospy.logerr('Error the element is not valid')
        elif frame == 'ee':
            if element.find('force') != -1:
                self.file_force_ee.write(line)
            else:
                rospy.logerr('Error the element is not valid')
        elif frame == 'world':
            if element.find('ee') != -1:
                self.file_ee_world.write(line)
            elif element.find('auv') != -1:
                self.file_auv_world.write(line)
            else:
                rospy.logerr('Error the element is not valid')
        else:
            rospy.logerr('Error the frame is not defined')

    def run(self):
        """
        Use the state machine to reproduce the action
        """
        rate = rospy.Rate(1.0/self.interval_time)
        # dmp_1 = LearningDmpReproductor(
        #     'Approach',
        #     'traj_auv_panel_first_aprox.txt',
        #     4,
        #     1.0,
        #     self.interval_time)
        # current_pose = [0.0 , 1.0, 4.5, 0.0]
        # current_vel = [0.0 , 0.0, 0.0, 0.0]
        # file_sim = open("traj_simulated.csv", 'w')
        # line = (repr(rospy.get_time()) + " " +
        #         repr(current_pose[0]) + " " +
        #         repr(current_pose[1]) + " " +
        #         repr(current_pose[2]) + " " +
        #         repr(current_pose[3]) + "\n")
        # file_sim.write(line)
        print 'Running!!!'
        #we search_for the initial state
        current_state = [
            st for st in self.list_of_states if st[0] == self.initial_state][0]
        rospy.loginfo('Starting Initial State ' + current_state[0])
        rospy.loginfo('Check frame ' + current_state[1][0])
        rospy.loginfo('Check type ' + current_state[1][1])
        rospy.loginfo('Check Condition ' + current_state[2][0][0])
        rospy.loginfo('Check State ' + current_state[2][0][1])
        prev_pose = np.array([])
        prev_vel = np.array([])
        init_pose = False
        des_pose = np.array([])
        des_vel = np.array([])
        rospy.loginfo('*******************************************************')
        rospy.loginfo('**************** Starting Hierarchical ****************')
        rospy.loginfo('**************** '
                      + str(current_state[0]) +
                      '*****************')
        while not rospy.is_shutdown():
            if self.enabled:
                if  not init_pose or not self.simulation:
                    current_pose  = self.obtain_current_pose(
                        current_state[1][0], current_state[1][1], prev_pose)
                else:
                    current_pose = des_pose
                    current_vel = des_vel
                #current_state
                if len(prev_vel) != 0:
                    #rospy.loginfo('Current pose ' + str(current_pose))
                    #rospy.loginfo('Prev pose ' + str(prev_pose))
                    current_vel = ((current_pose - prev_pose)
                                        / self.interval_time)
                    [des_pose, des_vel] = current_state[1][2].generateNewPose(
                        current_pose, current_vel)
                    #check end condition
                    if current_state[2][0][0] == 'finish_dmp':
                        if len(des_pose) != 0:
                            if self.simulation:
                                self.store_pose(current_pose,
                                                current_state[1][0],
                                                current_state[1][1])
                                current_pose = des_pose
                                init_pose = True
                            else:
                                pass
                        else:
                            #state has change next state
                            if current_state[2][0][1] == "end":
                                rospy.loginfo('****************
                                End Hierarchical ****************')
                                break
                            current_state = [
                                st for st in self.list_of_states
                                if st[0] == current_state[2][0][1]][0]
                            rospy.loginfo('**************** ' +
                                          ' Next State ' +
                                          ' *****************')
                            rospy.loginfo('**************** '
                                          + str(current_state[0]) +
                                          '*****************')
                            prev_pose = np.array([])
                            prev_vel = np.array([])
                            init_pose = False
                            current_pose = np.array([])
                            current_vel = np.array([])
                    elif current_state[2][0][0] == 'finish_all_dmp' :
                        if len(des_pose) != 0:
                            if self.simulation:
                                self.store_pose(current_pose,
                                                current_state[1][0],
                                                current_state[1][1])
                                current_pose = des_pose
                                init_pose = True
                            else:
                                pass
                        else:
                            #state has change next state
                            if current_state[2][0][1][0] == "end":
                                break
                            current_state = [
                                st for st in self.list_of_states
                                if st[0] == current_state[2][0][1]][0]
                            rospy.loginfo('**************** ' +
                                          ' Next State ' +
                                          ' *****************')
                            rospy.loginfo('**************** '
                                          + str(current_state[0]) +
                                          '*****************')
                            prev_pose = np.array([])
                            prev_vel = np.array([])
                            init_pose = False
                            current_pose = np.array([])
                            current_vel = np.array([])
                    else:
                        rospy.logerr('Not correct finish condition')
                    prev_pose = current_pose
                    prev_vel = current_vel
                elif len(prev_pose) == 0:
                    prev_pose = current_pose
                elif len(prev_vel) == 0:
                    prev_vel = (current_pose - prev_pose) / self.interval_time
                    prev_pose = current_pose

                # [desPos, desVel] = dmp_1.generateNewPose(
                #     current_pose, current_vel)
                # if len(desPos) == 0:
                #     break
                # current_pose = desPos
                # current_vel = desVel
                # line = (repr(rospy.get_time()) + " " +
                #         repr(current_pose[0]) + " " +
                #         repr(current_pose[1]) + " " +
                #         repr(current_pose[2]) + " " +
                #         repr(current_pose[3]) + "\n")
                # file_sim.write(line)
            rate.sleep()

        if self.simulation:
            #Files to store the trajectory
            self.file_ee_valve_0.close()
            self.file_ee_valve_1.close()
            self.file_ee_valve_2.close()
            self.file_ee_valve_3.close()
            self.file_ee_panel_centre.close()
            self.file_ee_auv.close()
            self.file_ee_world.close()

            self.file_auv_valve_0.close()
            self.file_auv_valve_1.close()
            self.file_auv_valve_2.close()
            self.file_auv_valve_3.close()
            self.file_auv_panel_centre.close()

            self.file_auv_world.close()

            self.file_force_ee.close()

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
