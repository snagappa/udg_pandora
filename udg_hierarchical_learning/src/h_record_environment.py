#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_hierarchical_learning')
import rospy

import numpy as np
#use to load the configuration function
from cola2_lib import cola2_ros_lib
#use to normalize the angle
from cola2_lib import cola2_lib
#import cola2_lib
#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the valve position
#from geometry_msgs.msg import PoseWithCovarianceStamped
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
#from geometry_msgs.msg import Quaternion
#from tf.transformations import euler_from_quaternion
#import to use mutex
import threading
import tf
import math


class HRecordEnvironment:

    def __init__(self, name):
        self.name = name
        self.get_config()
        rospy.loginfo('Configuration Loaded')
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

        self.initTime = 0.0
        self.init_element_auv = False
        self.init_element_ee = False
        self.init_element_force = False
        self.init_frame_valve_0 = False
        self.init_frame_valve_1 = False
        self.init_frame_valve_2 = False
        self.init_frame_valve_3 = False
        self.init_frame_panel_centre = False


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

        # Not needed ???
        # rospy.Subscriber(
        #     "/csip_e5_arm/joint_state",
        #     JointState,
        #     self.updateRollEndEffector,
        #     queue_size = 1)

        #Debug propose

    def get_config(self):
        param_dict = {'sample': '/hierarchical/record/sample',
                      'landmark_id': '/hierarchical/record/landmark_id',
                      'interval_time': '/hierarchical/record/interval_time',
                      'base_pose': '/arm_controller/base_pose',
                      'base_ori': '/arm_controller/base_ori'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)

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
            self.robotPose = odometry.pose.pose
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

    def store_pose_same_target_orgin(self, pose, new_origin, file):
        """
        This function convert the first pose and orientation in the world frame,
        to the frame passed as second argument. Finally it store it at the file
        passed as third parameter.
        @param pose: Position and Orientation of the element in world
        @type pose: Pose
        @param frame: Position and Orientation of the frame
        @type frame: Pose
        @param file: Position and Orientation of the element
        @type file: File
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
        new_pose.position.x = new_frame[1, 3]
        new_pose.position.x = new_frame[2, 3]

        euler = tf.transformations.euler_from_matrix(new_frame)

        line = (repr(rospy.get_time()) + " " +
                repr(new_frame[0,3]) + " " +
                repr(new_frame[1,3]) + " " +
                repr(new_frame[2,3]) + " " +
                repr(euler[0]) + " " +
                repr(euler[1]) + " " +
                repr(euler[2]) +"\n")
        file.write(line)

        return new_pose

    def store_pose_same_orgin(self, element, frame, ori, file):
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
        @param file: Position and Orientation of the element
        @type file: File
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
        new_pose.position.x = new_frame[1, 3]
        new_pose.position.x = new_frame[2, 3]

        euler = tf.transformations.euler_from_matrix(new_frame)

        line = (repr(rospy.get_time()) + " " +
                repr(new_frame[0,3]) + " " +
                repr(new_frame[1,3]) + " " +
                repr(new_frame[2,3]) + " " +
                repr(difference) + " " +
                repr(euler[0]) + " " +
                repr(euler[1]) + " " +
                repr(euler[2]) +"\n")
        file.write(line)

        return new_pose

    def store_force(self, force, file):
        """
        This function store the force and a pose stamped time.
        @param force: Force and Torque
        @type force: Wrench
        @param file: the file where has to be write the data
        @tyep file: File
        """
        line = (repr(rospy.get_time()) + " " +
                repr(force.force.x) + " " +
                repr(force.force.y) + " " +
                repr(force.force.z) + " " +
                repr(force.torque.x) + " " +
                repr(force.torque.y) + " " +
                repr(force.torque.z) +"\n")
        file.write(line)

    def store_pose(self, pose, file):
        """
        This function store the pose and the current time.
        @param force: Force and Torque
        @type force: Pose
        @param file: the file where has to be write the data
        @tyep file: File
        """

        euler = tf.transformations.euler_from_quaternion(
            [pose.orientation.x,
             pose.orientation.y,
             pose.orientation.z,
             pose.orientation.w])
        line = (repr(rospy.get_time()) + " " +
                repr(pose.position.x) + " " +
                repr(pose.position.y) + " " +
                repr(pose.position.z) + " " +
                repr(euler[0]) + " " +
                repr(euler[1]) + " " +
                repr(euler[2]) +"\n")
        file.write(line)


    def run(self):
        """
        This function is the main of the class and periodically calls the
        diferent method to write the position of the elements in different frames.
        """
        #open files the files
        file_ee_valve_0 = open("traj_ee_valve_0"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_ee_valve_1 = open("traj_ee_valve_1"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_ee_valve_2 = open("traj_ee_valve_2"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_ee_valve_3 = open("traj_ee_valve_3"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_ee_panel_centre = open("traj_ee_panel_centre"+ "_" +
                                    str(self.sample) + ".csv", 'w')
        file_ee_world = open("traj_ee_world"+ "_" +
                             str(self.sample) + ".csv", 'w')
        file_ee_auv = open("traj_ee_auv"+ "_" +
                           str(self.sample) + ".csv", 'w')

        file_auv_valve_0 = open("traj_auv_valve_0"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_auv_valve_1 = open("traj_auv_valve_1"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_auv_valve_2 = open("traj_auv_valve_2"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_auv_valve_3 = open("traj_auv_valve_3"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_auv_panel_centre = open("traj_auv_panel_centre"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_auv_world = open("traj_auv_world"+ "_" +
                               str(self.sample) + ".csv", 'w')
        file_force_world = open("force_world"+ "_" +
                               str(self.sample) + ".csv", 'w')

        rate = rospy.Rate(1.0/self.interval_time)
        rospy.loginfo('Recording')
        while not rospy.is_shutdown():
            if self.init_element_auv :
                if self.init_element_ee :
                    # Convert EE to the AUV frame
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

                    arm_base = self.store_pose_same_target_orgin(
                        self.element_ee, base_pose, file_ee_auv)
                    #world position
                    arm_world = self.store_pose_same_target_orgin(
                        self.element_ee, self.element_auv, file_ee_world)
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
                    if self.init_frame_valve_0 :
                        self.store_pose_same_orgin(
                            arm_world,
                            self.frame_valve_0,
                            self.frame_valve_0_handle,
                            file_ee_valve_0)
                    if self.init_frame_valve_1 :
                        self.store_pose_same_orgin(
                            arm_world,
                            self.frame_valve_1,
                            self.frame_valve_1_handle,
                            file_ee_valve_1)
                    if self.init_frame_valve_2 :
                        self.store_pose_same_orgin(
                            arm_world,
                            self.frame_valve_2,
                            self.frame_valve_2_handle,
                            file_ee_valve_2)
                    if self.init_frame_valve_3 :
                        self.store_pose_same_orgin(
                            arm_world,
                            self.frame_valve_3,
                            self.frame_valve_2_handle,
                            file_ee_valve_3)
                    if self.init_frame_panel_centre :
                        self.store_pose_same_orgin(
                            arm_world,
                            self.frame_panel_centre,
                            ori_panel,
                            file_ee_panel_centre)
            if self.init_element_ee :
                #world position
                self.store_pose(self.element_auv, file_auv_world)
                #strange work around with ori of the panel
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
                if self.init_frame_valve_0 :
                    self.store_pose_same_orgin(
                        self.element_auv,
                        self.frame_valve_0,
                        ori_work_around,
                        file_auv_valve_0)
                if self.init_frame_valve_1 :
                    self.store_pose_same_orgin(
                        self.element_auv,
                        self.frame_valve_1,
                        ori_work_around,
                        file_auv_valve_1)
                if self.init_frame_valve_2 :
                    self.store_pose_same_orgin(
                        self.element_auv,
                        self.frame_valve_2,
                        ori_work_around,
                        file_auv_valve_2)
                if self.init_frame_valve_3 :
                    self.store_pose_same_orgin(
                        self.element_auv,
                        self.frame_valve_3,
                        ori_work_around,
                        file_auv_valve_3)
                if self.init_frame_panel_centre :
                    self.store_pose_same_orgin(
                        self.element_auv,
                        self.frame_panel_centre,
                        ori_work_around,
                        file_auv_panel_centre)
            if self.init_element_force :
                self.store_force(self.element_force, file_force_world)
            rate.sleep()
        #close files
        file_ee_valve_0.close()
        file_ee_valve_1.close()
        file_ee_valve_2.close()
        file_ee_valve_3.close()
        file_ee_panel_centre.close()
        file_ee_world.close()
        file_ee_auv.close()

        file_auv_valve_0.close()
        file_auv_valve_1.close()
        file_auv_valve_2.close()
        file_auv_valve_3.close()
        file_auv_panel_centre.close()
        file_auv_world.close()

        file_force_world.close()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_hierarchical_learning", "h_record_environment.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate h_record_environment.yaml")

        rospy.init_node('h_record_environment')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        h_record_environment = HRecordEnvironment(rospy.get_name())
        h_record_environment.run()
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass
