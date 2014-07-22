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
from learning_pandora.msg import ValveTurningAction, ValveTurningGoal
from planning_msgs.msg import ActionDispatch
from planning_msgs.msg import ActionFeedback
from cola2_control.msg import WorldWaypointReqActionResult
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf
from diagnostic_msgs.msg import KeyValue
from pose_ekf_slam.msg import Map
from std_srvs.srv import Empty, EmptyRequest

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

        # Action Feedback/Dispatch publishers/subscriber
        self.pub_feedback = rospy.Publisher("/action_feedback",
                                            ActionFeedback)

        rospy.Subscriber("/action_dispatch",
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
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating client')
        

        # VALVE STATUS
        rospy.Subscriber("/valve_tracker/valve0",
                         PoseWithCovarianceStamped,
                         self.update_valve_0,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve1",
                         PoseWithCovarianceStamped,
                         self.update_valve_1,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve2",
                         PoseWithCovarianceStamped,
                         self.update_valve_2,
                         queue_size = 1)
        rospy.Subscriber("/valve_tracker/valve3",
                         PoseWithCovarianceStamped,
                         self.update_valve_3,
                         queue_size = 1)

        rospy.Subscriber("/pose_ekf_slam/update_landmark/panel_centre",
                         PoseWithCovarianceStamped,
                         self.update_panel_centre,
                         queue_size = 1)

        # CHECK PANEL
        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.update_ekf_panel_centre,
                         queue_size = 1)



        # TURN VALVE
        # TODO: Remember to start /learning/valve_turning_action and 
        #       uncomment this section!!!
        """
        self.turn_valve_action = actionlib.SimpleActionClient(
                                        '/learning/valve_turning_action',
                                        ValveTurningAction)
        rospy.loginfo("%s: wait for turn valve actionlib ...", self.name)
        self.turn_valve_action.wait_for_server()
        rospy.loginfo("%s: turn valve actionlib found!", self.name)
        """

        # CHAIN FOLLOW SERVICES 
        try:
            rospy.wait_for_service('/udg_pandora/enable_chain_planner', 10)
            self.enable_chain_planner_srv = rospy.ServiceProxy(
                                '/udg_pandora/enable_chain_planner', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating client')
            
        try:
            rospy.wait_for_service('/udg_pandora/disable_chain_planner', 10)
            self.disable_chain_planner_srv = rospy.ServiceProxy(
                                '/udg_pandora/disable_chain_planner', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating client')


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
        elif req.name == 'valve_state':
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
        elif req.name in ("check_panel", "observe"):
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
        else:
            rospy.logwarn('Invalid action name: ', req.name)


    # CLASS ACTIONS
    def __execute_turn_valve__(self, action_id, params):
        feedback = ActionFeedback()
        feedback.action_id = action_id
        feedback.status = 'action enabled'

        goal = ValveTurningGoal()
        goal.valve_id = int(params[0])
        goal.long_approach = False
        goal.desired_increment = float(params[1])
        self.turn_valve_action.send_goal(goal)

        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: turn valve action enabled', self.name)

        self.turn_valve_action.wait_for_result()
        feedback.status = 'action achieved'
        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: turn valve action achieved', self.name)


    def __execute_valve_state__(self, action_id):
        # Publish action enabled
        feedback = ActionFeedback()
        feedback.action_id = action_id
        feedback.status = 'action enabled'
        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: valve state action enabled', self.name)
        rospy.sleep(2.0)
        if rospy.Time.now().to_sec() - self.last_panel_update < 2.0:
            rospy.loginfo('%s: We are looking at the panel right now!', self.name)
            wait = action_id.duration - 2.0
            if wait < 0.0:
                wait = 0.0
            rospy.sleep(wait)
        else:
            rospy.loginfo('%s: Panel out of field of view.', self.name)


        # Publish action response
        ret = list()
        for i in range(4):
            if self.valve_covariance[i] > 0 and self.valve_covariance[i] < 1.0:
                element = KeyValue()
                element.key = 'valve_' + str(i) + '_angle'
                element.value = str(self.valve_orientation[i])
                ret.append(element)

                element1 = KeyValue()
                element1.key = 'valve_' + str(i) + '_in_panel'
                element1.value = '0'
                ret.append(element1)

        feedback.status = 'action achieved'
        feedback.information = ret
        rospy.loginfo('%s: valve state response: \n%s', self.name, feedback)
        self.pub_feedback.publish(feedback)


    def __execute_check_panel__(self, action_id):
        # Publish action enabled
        feedback = ActionFeedback()
        feedback.action_id = action_id
        feedback.status = 'action enabled'
        self.pub_feedback.publish(feedback)
        rospy.loginfo('%s: check panel action enabled', self.name)
        rospy.sleep(2.0)
        element = KeyValue()
        element.key = 'panel_0_in_fov'

        print 'current time: ', rospy.Time.now().to_sec()
        print 'last time we see the panel: ', self.last_panel_update
        if rospy.Time.now().to_sec() - self.last_panel_update < 2.0:
            rospy.loginfo('%s: We are looking at the panel right now!', self.name)
            wait = action_id.duration - 2.0
            if wait < 0.0:
                wait = 0.0
            element.value = 'true'
            rospy.sleep(wait)
        else:
            element.value = 'false'
            rospy.loginfo('%s: Panel out of field of view.', self.name)

        ret = list()
        ret.append(element)

        # Publish action response
        if self.ekf_panel_centre != None:
            element1 = KeyValue()
            element1.key = 'panel_0_position'
            position = [self.ekf_panel_centre.x,
                        self.ekf_panel_centre.y,
                        self.ekf_panel_centre.z,
			self.ekf_panel_yaw]
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
        angle = tf.transformations.euler_from_quaternion(
                                    [data.pose.pose.orientation.x,
                                     data.pose.pose.orientation.y,
                                     data.pose.pose.orientation.z,
                                     data.pose.pose.orientation.w])
        self.valve_orientation[valve_id] = angle[2]
        self.valve_covariance[valve_id] = data.pose.covariance[35]
        self.valve_pose[valve_id][0] = data.pose.pose.position.x
        self.valve_pose[valve_id][1] = data.pose.pose.position.y
        self.valve_pose[valve_id][2] = data.pose.pose.position.z

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
        print '**'


    def update_ekf_panel_centre(self, data):
        if len(data.landmark) > 0:
            self.ekf_panel_centre = data.landmark[0].pose.pose.position
            angle = tf.transformations.euler_from_quaternion(
                                    [data.landmark[0].pose.pose.orientation.x,
                                     data.landmark[0].pose.pose.orientation.y,
                                     data.landmark[0].pose.pose.orientation.z,
                                     data.landmark[0].pose.pose.orientation.w])
            self.ekf_panel_yaw = angle[1]


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
