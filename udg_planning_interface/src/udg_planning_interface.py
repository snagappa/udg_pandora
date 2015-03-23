#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4, 2013

@author: narcis palomeras
"""

# ROS imports
import roslib
roslib.load_manifest('udg_planning_interface')
import rospy
import actionlib
from cola2_control.srv import GotoWithYaw, GotoWithYawRequest
from cola2_safety.srv import DigitalOutput
from learning_pandora.msg import ValveTurningAction, ValveTurningGoal
from planning_msgs.msg import ActionDispatch
from planning_msgs.msg import ActionFeedback
from cola2_control.msg import WorldWaypointReqActionResult
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf
from diagnostic_msgs.msg import KeyValue
from pose_ekf_slam.msg import Map
from std_srvs.srv import Empty, EmptyRequest
from std_msgs.msg import Float64
from cola2_control.srv import EFPose, ValveBlocked, ValveBlockedResponse

import numpy as np

class PlanningInterface(object):
    def __init__(self, name):
        self.name = name
        self.goto_action_id = -1
        self.valve_orientation = [0.0, 0.0, 0.0, 0.0]
        self.valve_covariance = [-1.0, -1.0, -1.0, -1.0]
        self.valve_pose = [[0.0,0.0,0.0],
                           [0.0,0.0,0.0],
                           [0.0,0.0,0.0],
                           [0.0,0.0,0.0]]
        self.last_panel_update = rospy.Time.now().to_sec() - 10
        self.ekf_panel_centre = None
        self.is_valve_blocked = [False, False, False, False]

        # Action Feedback/Dispatch publishers/subscriber
        self.pub_feedback = rospy.Publisher("/planning_system/action_feedback",
                                            ActionFeedback)

        rospy.Subscriber("/planning_system/action_dispatch",
                         ActionDispatch,
                         self.dispatch_action,
                         queue_size = 1)

        # GOTO
        rospy.Subscriber("/absolute_movement/result",
                         WorldWaypointReqActionResult,
                         self.goto_result,
                         queue_size = 1)

        try:
            rospy.wait_for_service('/cola2_control/goto_holonomic', 20)
            self.goto_holonomic_srv = rospy.ServiceProxy(
                                '/cola2_control/goto_holonomic', GotoWithYaw)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (goto_holonomic)', name)
            rospy.signal_shutdown('Error creating client')


        # VALVE STATUS
        # rospy.Subscriber("/valve_tracker/valve_0_ori",
        #                  Float64,
        #                  self.update_valve_0,
        #                  queue_size = 1)
        # rospy.Subscriber("/valve_tracker/valve_1_ori",
        #                  Float64,
        #                  self.update_valve_1,
        #                  queue_size = 1)
        # rospy.Subscriber("/valve_tracker/valve_2_ori",
        #                  Float64,
        #                  self.update_valve_2,
        #                  queue_size = 1)
        # rospy.Subscriber("/valve_tracker/valve_3_ori",
        #                  Float64,
        #                  self.update_valve_3,
        #                  queue_size = 1)

        rospy.Subscriber("/valve_tracker/valve_0_ass_ori",
                         Float64,
                         self.update_valve_0,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve_1_ass_ori",
                         Float64,
                         self.update_valve_1,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve_2_ass_ori",
                         Float64,
                         self.update_valve_2,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve_3_ass_ori",
                         Float64,
                         self.update_valve_3,
                         queue_size = 1)


        rospy.Subscriber("/pose_ekf_slam/landmark_update/panel_centre",
                         PoseWithCovarianceStamped,
                         self.update_panel_centre,
                         queue_size = 1)

        # CHECK PANEL
        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.update_ekf_panel_centre,
                         queue_size = 1)

        # Lights On
        try:
            rospy.wait_for_service('/digital_output', 20)
            self.digital_output_srv = rospy.ServiceProxy(
                                '/digital_output', DigitalOutput)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (digital_output)', name)
            rospy.signal_shutdown('Error creating client')


        # TURN VALVE
        # TODO: Remember to start /learning/valve_turning_action and
        #       uncomment this section!!!
        """ self.turn_valve_action = actionlib.SimpleActionClient('/learning/valve_turning_action', ValveTurningAction)
        rospy.loginfo("%s: wait for turn valve actionlib ...", self.name)
        self.turn_valve_action.wait_for_server()
        rospy.loginfo("%s: turn valve actionlib found!", self.name)

        try:
            rospy.wait_for_service('/valve_tracker/disable_update_valve_orientation', 10)
            self.disable_valve_orientation_srv = rospy.ServiceProxy("/valve_tracker/disable_update_valve_orientation", Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (valve orientation disable)', name)
            rospy.signal_shutdown('Error creating client')

        try:
            rospy.wait_for_service('/valve_tracker/enable_update_valve_orientation', 10)
            self.enable_valve_orientation_srv = rospy.ServiceProxy("/valve_tracker/enable_update_valve_orientation", Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (valve orientation enable)', name)
            rospy.signal_shutdown('Error creating client')
        """

        # CHAIN FOLLOW SERVICES
        try:
            rospy.wait_for_service('/udg_pandora/enable_chain_planner', 10)
            self.enable_chain_planner_srv = rospy.ServiceProxy(
                                '/udg_pandora/enable_chain_planner', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (chain planner enable)', name)
            rospy.enable_valve_orientation_srv = rospy.signal_shutdown('Error creating client')
        try:
            rospy.wait_for_service('/udg_pandora/disable_chain_planner', 10)
            self.disable_chain_planner_srv = rospy.ServiceProxy(
                                '/udg_pandora/disable_chain_planner', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client. (chain planner disable)', name)
            rospy.signal_shutdown('Error creating client')

        # CALIBRATE ARM
        """try:
            rospy.wait_for_service('/cola2_control/arm_calibration', 10)
            self.enable_arm_calibration_srv = rospy.ServiceProxy(
                '/cola2_control/arm_calibration', Empty)
        except rospy.exceptions.ROSException:
           rospy.logerr('%s, Error creating client. (Arm Calibration enable)', name)
           rospy.signal_shutdown('Error creating client')
        """
        
        # RESET LANDMARKS
        try:
            rospy.wait_for_service('/cola2_navigation/reset_landmarks', 10)
            self.reset_landmarks_srv = rospy.ServiceProxy(
                '/cola2_navigation/reset_landmarks', Empty)
        except rospy.exceptions.ROSException:
           rospy.logerr('%s, Error creating client. (Reset landmarks enable)', name)
           rospy.signal_shutdown('Error creating client')
           
        # Enable Keep position
        try:
            rospy.wait_for_service('/cola2_control/enable_keep_position_g500', 10)
            self.enable_keep_pose_srv = rospy.ServiceProxy(
                '/cola2_control/enable_keep_position_g500', Empty)
        except rospy.exceptions.ROSException:
           rospy.logerr('%s, Error creating client. (Enable_keep_position enable)', name)
           rospy.signal_shutdown('Error creating client')

        # Disable Keep position
        try:
            rospy.wait_for_service('/cola2_control/disable_keep_position', 10)
            self.disable_keep_pose_srv = rospy.ServiceProxy(
                '/cola2_control/disable_keep_position', Empty)
        except rospy.exceptions.ROSException:
           rospy.logerr('%s, Error creating client. (Enable_keep_position disable)', name)
           rospy.signal_shutdown('Error creating client')


        self.enable_srv = rospy.Service(
            '/valve_blocked',
            ValveBlocked,
            self.valve_blocked_srv)

        # By default turn the lights on
        self.digital_output_srv.call(13, True)

    def dispatch_action(self, req):
        if req.name == 'goto':
            rospy.loginfo("%s: Received goto action.", self.name)
            param_list = ['north', 'east', 'depth', 'yaw']
            params = __get_params__(req.parameters, param_list)
            rospy.loginfo("%s: Params: %s", self.name, params)
            if len(params) != 4:
                rospy.logwarn('some of these params have not been found: %s',
                              param_list)
            else:
                self.__execute_goto__(req.action_id, params)
        elif req.name in ("valve_state", "examine_panel"):
            rospy.loginfo("%s: Received valve_state action.", self.name)
            self.__execute_valve_state__(req.action_id)
        elif req.name == 'turn_valve':
            rospy.loginfo("%s: Received turn_valve action.", self.name)
            param_list = ['valve_id', 'desired_increment']
            params = __get_params__(req.parameters, param_list)
            rospy.loginfo("%s: Params: %s", self.name, params)
            if len(params) != 2:
                rospy.logwarn('some of these params have not been found: %s',
                              param_list)
            else:
                self.__execute_turn_valve__(req.action_id, params)
        elif req.name in ("check_panel", "observe", "observe_inspection_point"):
            rospy.loginfo("%s: Received check_panel action.", self.name)
            self.__execute_check_panel__(req.action_id)
        elif req.name == 'enable_chain_follow':
            rospy.loginfo("%s: Received enable_chain_follow action.",
                          self.name)
            self.enable_chain_planner_srv(EmptyRequest())
            feedback = ActionFeedback()
            feedback.action_id = req.action_id
            feedback.status = "action enabled"
            self.pub_feedback.publish(feedback)
        elif req.name == 'disable_chain_follow':
            rospy.loginfo("%s: Received disable_chain_follow action.",
                          self.name)
            self.disable_chain_planner_srv(EmptyRequest())
            feedback = ActionFeedback()
            feedback.action_id = req.action_id
            feedback.status = "action enabled"
            self.pub_feedback.publish(feedback)
        elif req.name == 'recalibrate_arm':
            rospy.loginfo("%s: Received recalibrate_arm action.",
                          self.name)
            feedback = ActionFeedback()
            feedback.action_id = req.action_id
            feedback.status = "action enabled"
            self.pub_feedback.publish(feedback)
            self.enable_keep_pose_srv(EmptyRequest())
            self.disable_valve_orientation_srv(EmptyRequest())
            self.digital_output_srv.call(13, False)
            self.enable_arm_calibration_srv(EmptyRequest())
            self.digital_output_srv.call(13, True)
            self.enable_valve_orientation_srv(EmptyRequest())
            self.disable_keep_pose_srv(EmptyRequest())
            # fold_arm_srv = rospy.ServiceProxy('/cola2_control/setPoseEF', EFPose)
            # value = fold_arm_srv([0.45, 0.0, 0.11, 0.0, 0.0, 0.0 ])
            #rospy.sleep(30.0)
            # feedback = ActionFeedback()
            # feedback.action_id = req.action_id
            feedback.status = "action achieved"
            self.pub_feedback.publish(feedback)

        elif req.name == 'reset_landmarks':
            feedback = ActionFeedback()
            feedback.action_id = req.action_id
            feedback.status = "action enabled"
            self.pub_feedback.publish(feedback)
            rospy.loginfo("%s: Received reset_landmarks action.",
                          self.name)
            self.reset_landmarks_srv(EmptyRequest())
            feedback.status = "action achieved"
            self.pub_feedback.publish(feedback)

        elif req.name == 'cancel_action':
            rospy.loginfo("%s: Received cancel_action request.", self.name)
            self.turn_valve_action.cancel_goal()
            feedback = ActionFeedback()
            feedback.action_id = req.action_id
            feedback.status = "action failed"
            self.pub_feedback.publish(feedback)
        else:
            rospy.logwarn('Invalid action name: %s', req.name)


    # CLASS ACTIONS
    def __execute_turn_valve__(self, action_id, params):
        if float(params[1]) != 0.0 :
            feedback = ActionFeedback()
            feedback.action_id = action_id
            feedback.status = 'action enabled'

            # disable valve angle update
            self.disable_valve_orientation_srv(EmptyRequest())

            goal = ValveTurningGoal()
            goal.valve_id = int(params[0])
            goal.long_approach = False
            desired_increment = float(params[1])
            desired_increment = np.sign(desired_increment)*0.2 + desired_increment
            goal.desired_increment = -1.0*desired_increment
            self.turn_valve_action.send_goal(goal)

            self.pub_feedback.publish(feedback)
            rospy.loginfo('%s: turn valve action enabled', self.name)

            self.turn_valve_action.wait_for_result()
            action_result = self.turn_valve_action.get_result()
            # if action_result.valve_turned:
            #     feedback.status = 'action achieved'
            # else:
            #     feedback.status = 'action failed'
            #     if action_result.error_code == 1:
            #         # Valve is blocked
            #         element = KeyValue()
            #         element.key = 'valve_' + params[0] + '_state'
            #         element.value = "valve_blocked"
            #         feedback.information.append(element)

            if not self.is_valve_blocked[int(params[0])]:
                feedback.status = 'action achieved'
            else:
                feedback.status = 'action failed'
                #if action_result.error_code == 1:
                    # Valve is blocked
                element = KeyValue()
                element.key = 'valve_' + params[0] + '_state'
                element.value = "valve_blocked"
                feedback.information.append(element)
            self.pub_feedback.publish(feedback)
            rospy.loginfo('%s: turn valve action finished', self.name)

            # enable valve angle update
            self.enable_valve_orientation_srv(EmptyRequest())
        else:
            feedback = ActionFeedback()
            feedback.action_id = action_id
            feedback.status = 'action achieved'
            self.pub_feedback.publish(feedback)

    def __execute_valve_state__(self, action_id):
        # Publish action enabled
        feedback = ActionFeedback()
        feedback.action_id = action_id
        feedback.status = 'action enabled'
        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: valve state action enabled', self.name)
        #rospy.sleep(10.0)
        rospy.sleep(3.0)
        if rospy.Time.now().to_sec() - self.last_panel_update < 8.0:
            rospy.loginfo('%s: We are looking at the panel right now!', self.name)

            # Publish action response
            ret = list()
            for i in range(4):
                if self.valve_covariance[i] > 0 and self.valve_covariance[i] < 1.0:
                    element = KeyValue()
                    element.key = 'valve_' + str(i) + '_angle'
                    element.value = str(self.valve_orientation[i])
                    ret.append(element)

                    # If we have more than one panel we indicated the panel id (now is always 0)
                    element1 = KeyValue()
                    element1.key = 'valve_' + str(i) + '_in_panel'
                    element1.value = '0'
                    ret.append(element1)

            feedback.status = 'action achieved'
            feedback.information = ret
            rospy.loginfo('%s: valve state response: \n%s', self.name, feedback)
            self.pub_feedback.publish(feedback)
        else:
            rospy.loginfo('%s: Panel out of field of view.', self.name)
            feedback.status = 'action failed'
            element = KeyValue()
            element.key = 'panel_0_state'
            element.value = 'panel_missing'
            feedback.information.append(element)
            self.pub_feedback.publish(feedback)


    def __execute_check_panel__(self, action_id):

        # Publish action enabled
        feedback = ActionFeedback()
        feedback.action_id = action_id
        feedback.status = 'action enabled'
        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: check panel action enabled', self.name)

        # forget old panel location
        self.ekf_panel_centre = None

        rospy.sleep(2.0)
        element = KeyValue()
        element.key = 'panel_0_in_fov'

        print 'current time: ', rospy.Time.now().to_sec()
        print 'last time we see the panel: ', self.last_panel_update
        rospy.sleep(8)
        if rospy.Time.now().to_sec() - self.last_panel_update < 2.0:
            element.value = 'true'
        else:
            element.value = 'false'
            rospy.loginfo('%s: Panel out of field of view.', self.name)

        ret = list()
        rospy.wait_for_service('/cola2_control/goto_holonomic', 20)
        ret.append(element)

        # Publish action response
        if self.ekf_panel_centre != None:
            element1 = KeyValue()
            element1.key = 'panel_0_position'
            position = [self.ekf_panel_centre.position.x,
                        self.ekf_panel_centre.position.y,
                        self.ekf_panel_centre.position.z,
                        self.ekf_panel_centre.orientation.x,
                        self.ekf_panel_centre.orientation.y,
                        self.ekf_panel_centre.orientation.z,
                        self.ekf_panel_centre.orientation.w]
            element1.value = str(position)
            ret.append(element1)

        feedback.status = 'action achieved'
        feedback.information = ret
        rospy.loginfo('%s: check panel response: \n%s', self.name, feedback)
        self.pub_feedback.publish(feedback)


    def __execute_goto__(self, action_id, params):
        goto_req = GotoWithYawRequest()
        goto_req.north_lat = float(params[0])
        goto_req.east_lon = float(params[1])
        goto_req.z = float(params[2])
        goto_req.altitude_mode = False
        goto_req.yaw = float(params[3])
        goto_req.tolerance = 0.3

        # TODO: try-except???
        rospy.loginfo("%s: Execute goto action.", self.name)
        res = self.goto_holonomic_srv(goto_req)

        feedback = ActionFeedback()
        feedback.action_id = action_id
        attempts = 0
        while not res.attempted and attempts < 10:
            attempts = attempts + 1
            rospy.logwarn('Error starting goto_holonomic service. Trying again....')
            rospy.sleep(1.0)
            res = self.goto_holonomic_srv(goto_req)
        if  not res.attempted:
            rospy.logwarn('Error starting goto_holonomic service.')
            feedback.status = "error enabling action"
        else:
            feedback.status = "action enabled"
            self.goto_action_id = action_id
        self.pub_feedback.publish(feedback)


    # CLASS AUXILIAR METHODS
    def update_valve_0(self, data):
        self.set_valve_info(0, data)

    def update_valve_1(self, data):
        self.set_valve_info(1, data)

    def update_valve_2(self, data):
        self.set_valve_info(2, data)

    def update_valve_3(self, data):
        self.set_valve_info(3, data)

    def set_valve_info(self, valve_id, data):
        self.valve_orientation[valve_id] = data.data
        self.valve_covariance[valve_id] = 0.1

    def goto_result(self, data):
        print 'goto_result: \n', data
        if self.goto_action_id >= 0:
            feedback = ActionFeedback()
            feedback.action_id = self.goto_action_id
            feedback.status = "action achieved"
            self.goto_action_id = -1
            rospy.loginfo("%s: Send feedback: \n%s", self.name, feedback)
            self.pub_feedback.publish(feedback)

    def update_panel_centre(self, data):
        # Save last time that the panel was in the field of view
        self.last_panel_update = rospy.Time.now().to_sec()


    def update_ekf_panel_centre(self, data):
        if len(data.landmark) > 0:
            self.ekf_panel_centre = data.landmark[0].pose.pose


    def valve_blocked_srv(self, req):
        if req.valve_id >= 4 or req.valve_id < 0:
            return ValveBlockedResponse(False)
        self.is_valve_blocked[req.valve_id] = req.valve_blocked
        rospy.loginfo('Valve Blocked ' + str(self.is_valve_blocked))
        return ValveBlockedResponse(True)
# PRIVATE AUXILIAR FUNCTIONS

def  __get_params__(key_value, param_list):
    ret = list()
    for param in param_list:
        found = False
        i = 0
        while not found and i < len(key_value):
            if key_value[i].key == param:
                ret.append(key_value[i].value)
                found = True
            else:
                i = i + 1
        # If a key is not found returns an empty list
        if i == len(key_value):
            return []
    # If everything is fine returns a list with all the values
    return ret


def __set_params__(param_list):
    ret = list()
    for key in param_list:
        element = KeyValue()
        element.key = key
        element.value = param_list[key]
        ret.append(element)
    return ret


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('udg_planning_interface')
        planning_interface = PlanningInterface(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass