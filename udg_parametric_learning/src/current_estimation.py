#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_parametric_learning')
import rospy

import numpy as np

#use to load the configuration function
from cola2_lib import cola2_ros_lib

from auv_msgs.msg import BodyForceReq

from geometry_msgs.msg import Twist, Pose

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from std_srvs.srv import Empty, EmptyResponse
from udg_parametric_learning.srv import StaticCurrent, StaticCurrentResponse

from cola2_control.srv import StareLandmark, StareLandmarkRequest

class CurrentEstimation:

    def __init__(self, name):
        self.name = name
        self.enable = False
        self.current_force = BodyForceReq()
        # Default parameters
        self.rate = 10.0
        self.time_analize = 20.0
        self.time_window = 10.0
        self.regular_force = []
        self.stare_landmark_enabled = False
        self.get_config()
        #service
        self.enable_srv = rospy.Service(
            '/current_estimator/enable_current_estimation',
            Empty,
            self.enable_srv)

        self.disable_srv = rospy.Service(
            '/current_estimator/disable_current_estimation',
            Empty,
            self.disable_srv)

        self.current_static_srv = rospy.Service(
            '/current_estimator/static_estimation',
            StaticCurrent,
            self.compute_static_current_srv)

        #subscriber
        rospy.Subscriber('/cola2_control/merged_body_force_req',
                         BodyForceReq,
                         self.update_auv_force,
                         queue_size = 1)

        #publisher
        self.pub_current_estimation = rospy.Publisher(
            "/current_estimation/current_vector", Twist)

        #get clients
        if self.stare_landmark_enabled:
            # TODO Use got to Landmark
            #rospy.wait_for_service('/cola2_control/goto_landmark')
            rospy.wait_for_service('/cola2_control/enable_stare_landmark')
            rospy.wait_for_service('/cola2_control/disable_stare_landmark')
            # self.keep_position_enable = rospy.ServiceProxy(
            #     '/cola2_control/goto_landmark',
            #     StareLandmark)
            self.keep_position_enable = rospy.ServiceProxy(
                '/cola2_control/enable_stare_landmark',
                StareLandmark)
            self.keep_position_disable = rospy.ServiceProxy(
                '/cola2_control/disable_stare_landmark',
                Empty)
        else:
            rospy.wait_for_service('/cola2_control/enable_keep_position_g500')
            rospy.wait_for_service('/cola2_control/disable_keep_position')
            self.keep_position_enable = rospy.ServiceProxy(
                '/cola2_control/enable_keep_position_g500',
                Empty)
            self.keep_position_disable = rospy.ServiceProxy(
                '/cola2_control/disable_keep_position',
                Empty)

        if not self.regular_force:
            rospy.loginfo('Computing the regular force')
            self.compute_regular_force()

        rospy.loginfo('Node initialized Correctly')

    def get_config(self):
        '''
        Load the file parameters.
        '''
        param_dict = {'rate': 'current_estimation/rate',
                      'time_analize': 'current_estimation/time_analize',
                      'regular_force': 'current_estimation/regular_force',
                      'stare_landmark_enabled': 'current_estimation/stare_landmark_enabled',
                      'stare_landmark_id': 'current_estimation/stare_landmark_id',
                      'stare_landmark_offset': 'current_estimation/stare_landmark_offset',
                      'stare_landmark_tolerance': 'current_estimation/stare_landmark_tolerance',
                      'stare_landmark_keep_pose': 'current_estimation/stare_landmark_keep_pose'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Time analize ' + str(self.time_analize))

    def enable_srv(self, req):
        self.enable = True
        rospy.loginfo('%s Enabled', self.name)
        return EmptyResponse()

    def disable_srv(self, req):
        self.enable = False
        rospy.loginfo('%s Disabled', self.name)
        return EmptyResponse()

    def update_auv_force(self, msg):
        '''
        Update the force computed
        '''
        self.current_force = msg

    def compute_regular_force(self):
        '''
        keep the current position in a place where there are not currents
        during the defined time and compute the regular force
        '''
        rate = rospy.Rate(self.rate)
        # Call the service
        if self.stare_landmark_enabled:
            request = StareLandmarkRequest()
            request.landmark_id = self.stare_landmark_id
            # offset type is geometry_msgs/Pose
            request.offset = Pose()
            request.offset.position.x = self.stare_landmark_offset[0]
            request.offset.position.y = self.stare_landmark_offset[1]
            request.offset.position.z = self.stare_landmark_offset[2]
            quaternion = quaternion_from_euler(self.stare_landmark_offset[3],
                                               self.stare_landmark_offset[4],
                                               self.stare_landmark_offset[5])
            request.offset.orientation.x = quaternion[0]
            request.offset.orientation.y = quaternion[1]
            request.offset.orientation.z = quaternion[2]
            request.offset.orientation.w = quaternion[3]
            request.tolerance = self.stare_landmark_tolerance
            request.keep_pose = self.stare_landmark_keep_pose
            answer = self.keep_position_enable.call(request)
        else:
            self.keep_position_enable.call()
        # take the init time
        init_time = rospy.get_time()
        force_vector = []
        counter = 0.0
        while not rospy.is_shutdown() and (rospy.get_time() - init_time) <= self.time_analize :
            force_vector.append(self.current_force.wrench.force.x)
            force_vector.append(self.current_force.wrench.force.y)
            force_vector.append(self.current_force.wrench.force.z)
            force_vector.append(self.current_force.wrench.torque.x)
            force_vector.append(self.current_force.wrench.torque.y)
            force_vector.append(self.current_force.wrench.torque.z)
            counter += 1.0
            rospy.loginfo('Time ' + str(rospy.get_time() - init_time) + ' Time_analize ' + str(self.time_analize))
            rate.sleep()
        self.keep_position_disable.call()
        matrix = np.reshape(force_vector, (counter,6))
        avg_force = [0.0] * 6
        for i in range(6):
            avg_force[i] = np.sum(matrix[:,i]) / counter
        self.regular_force = avg_force

    def compute_static_current_srv(self, req):
        '''
        Loop sending a command to keep the standard position compare the needed
        force with the previous average force and return it as a vector.
        '''
        rate = rospy.Rate(self.rate)
        # Call the service
        if self.stare_landmark_enabled:
            request = StareLandmarkRequest()
            request.landmark_id = self.stare_landmark_id
            # offset type is geometry_msgs/Pose
            request.offset = Pose()
            request.offset.position.x = self.stare_landmark_offset[0]
            request.offset.position.y = self.stare_landmark_offset[1]
            request.offset.position.z = self.stare_landmark_offset[2]
            quaternion = quaternion_from_euler(self.stare_landmark_offset[3:6])
            request.offset.orientation.x = quaternion[0]
            request.offset.orientation.y = quaternion[1]
            request.offset.orientation.z = quaternion[2]
            request.offset.orientation.w = quaternion[3]
            request.tolerance = self.stare_landmark_tolerance
            request.keep_pose = self.stare_landmark_keep_pose
            answer = self.keep_position_enable.call(request)
        else:
            self.keep_position_enable.call()
        # take the init time
        init_time = rospy.get_time()
        force_vector = []
        counter = 0.0
        self.enable = False
        while not rospy.is_shutdown() and (rospy.get_time() - init_time) <= self.time_analize :
            force_vector.append(
                self.current_force.wrench.force.x-self.regular_force[0])
            force_vector.append(
                self.current_force.wrench.force.y-self.regular_force[1])
            force_vector.append(
                self.current_force.wrench.force.z-self.regular_force[2])
            force_vector.append(
                self.current_force.wrench.torque.x-self.regular_force[3])
            force_vector.append(
                self.current_force.wrench.torque.y-self.regular_force[4])
            force_vector.append(
                self.current_force.wrench.torque.z-self.regular_force[5])
            counter += 1.0
            rate.sleep()
        self.keep_position_disable.call()
        matrix = np.reshape(force_vector, (counter,6))
        avg_force = [0.0] * 6
        for i in range(6):
            avg_force[i] = np.sum(matrix[:,i]) / counter
        resp = StaticCurrentResponse()
        #resp.current_estimation = np.asarray(avg_force)
        resp.current_estimation = avg_force
        return resp

    def compute_current_dynamically(self):
        '''
        Compute in a constant way the values of the vector in this case we will use a
        window of 10 seconds to make it more robust against strange detections
        '''
        initialized = False
        rate = rospy.Rate(self.rate)
        counter = 0.0
        force_vector = []
        matrix = []
        avg_force = [0.0] * 6
        init_time = -999.0
        while not rospy.is_shutdown():
            if self.enable:
                if not initialized and ((rospy.get_time() - init_time) <= self.time_window or init_time == -999.0 ):
                    force_vector.append(
                        self.current_force.wrench.force.x-self.regular_force[0])
                    force_vector.append(
                        self.current_force.wrench.force.y-self.regular_force[1])
                    force_vector.append(
                        self.current_force.wrench.force.z-self.regular_force[2])
                    force_vector.append(
                        self.current_force.wrench.torque.x-self.regular_force[3])
                    force_vector.append(
                        self.current_force.wrench.torque.y-self.regular_force[4])
                    force_vector.append(
                        self.current_force.wrench.torque.z-self.regular_force[5])
                    if init_time == -999.0:
                        init_time = rospy.get_time()

                    counter += 1.0
                    #rospy.loginfo('Initializing time ' + str(rospy.get_time() - init_time) + 'Window Time ' + str(self.time_window) )
                elif not initialized and (rospy.get_time() - init_time) > self.time_window :
                    initialized = True
                    #reshape
                    matrix = np.reshape(force_vector, (counter,6))
                    avg_force = [0.0] * 6
                    for i in range(6):
                        avg_force[i] = np.sum(matrix[:,i]) / counter
                    msg = Twist()
                    msg.linear.x = avg_force[0]
                    msg.linear.y = avg_force[1]
                    msg.linear.z = avg_force[2]
                    msg.angular.x = avg_force[3]
                    msg.angular.y = avg_force[4]
                    msg.angular.z = avg_force[5]
                    self.pub_current_estimation.publish(msg)
                    rospy.loginfo('Start Publishing')
                elif initialized:
                    matrix[0:-1-1,:] = matrix[1:-1,:]
                    matrix[-1, 0] = (
                        self.current_force.wrench.force.x-self.regular_force[0])
                    matrix[-1, 1] = (
                        self.current_force.wrench.force.y-self.regular_force[1])
                    matrix[-1, 2] = (
                        self.current_force.wrench.force.z-self.regular_force[2])
                    matrix[-1, 3] = (
                        self.current_force.wrench.torque.x-self.regular_force[3])
                    matrix[-1, 4] = (
                        self.current_force.wrench.torque.y-self.regular_force[4])
                    matrix[-1, 5] = (
                        self.current_force.wrench.torque.z-self.regular_force[5])
                    avg_force = [0.0] * 6
                    # rospy.loginfo( 'Diff force ' + str(self.current_force.wrench.force.x-self.regular_force[0]) + ' ' + str(self.current_force.wrench.force.y-self.regular_force[1]) + ' ' + str(self.current_force.wrench.force.z-self.regular_force[2]) + ' ' + str(self.current_force.wrench.torque.x-self.regular_force[3]) + ' ' + str(self.current_force.wrench.torque.y-self.regular_force[4]) + ' ' + str(self.current_force.wrench.torque.x-self.regular_force[5]))
                    for i in range(6):
                        avg_force[i] = np.sum(matrix[:,i]) / counter
                    msg = Twist()
                    msg.linear.x = avg_force[0]
                    msg.linear.y = avg_force[1]
                    msg.linear.z = avg_force[2]
                    msg.angular.x = avg_force[3]
                    msg.angular.y = avg_force[4]
                    msg.angular.z = avg_force[5]
                    self.pub_current_estimation.publish(msg)
            else:
                init_time = -999.0
                counter = 0.0
                force_vector = []
                matrix = []
                avg_force = [0.0] * 6
                initialized = False
            rate.sleep()

if __name__ == '__main__':
    try:
        #Load the configuration file
        import subprocess
        config_file_list = roslib.packages.find_resource(
            "udg_parametric_learning", "current_estimation.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            rospy.logerr("Could not locate current_estimation.yaml")

        rospy.init_node('current_estimation')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        currentEstimation = CurrentEstimation(rospy.get_name())
        currentEstimation.compute_current_dynamically()
    except rospy.ROSInterruptException:
        pass
