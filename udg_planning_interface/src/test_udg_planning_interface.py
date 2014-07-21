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
from planning_msgs.msg import ActionDispatch
from planning_msgs.msg import ActionFeedback
from diagnostic_msgs.msg import KeyValue

class PlanningInterface(object):
    def __init__(self, name):
        self.name = name
        self.action_id = 0
        self.is_action_enabled = False
        self.is_action_running = False

        # Create publishers
        self.pub_action_dispatch = rospy.Publisher("/action_dispatch",
                                            ActionDispatch)

        # Create Subscribers
        rospy.Subscriber("/action_feedback",
                         ActionFeedback,
                         self.action_feedback,
                         queue_size = 1)


    def action_feedback(self, req):
        rospy.loginfo('%s: Received feedback:\n %s', self.name, req)
        if req.status == "action enabled":
            self.is_action_enabled = True
        elif req.status == "action achieved":
            self.is_action_running = False

    def execute_goto(self, x, y, z, yaw):
        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'goto'
            param_list = {'north': str(x),
                          'east': str(y),
                          'depth': str(z),
                          'yaw': str(yaw)}
            action.parameters = __set_params__(param_list)
            rospy.loginfo("%s: Execute action goto: %s", self.name, param_list)
            while not self.is_action_enabled:
                # print 'Action: \n', action
                self.pub_action_dispatch.publish(action)
                rospy.sleep(1.0)
            self.is_action_enabled = False


    def execute_valve_state(self):
        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'valve_state'

            while not self.is_action_enabled:
                print 'Action: \n', action
                self.pub_action_dispatch.publish(action)
                rospy.sleep(1.0)
            self.is_action_enabled = False


    def execute_check_panel(self):
        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'check_panel'

            while not self.is_action_enabled:
                print 'Action: \n', action
                self.pub_action_dispatch.publish(action)
                rospy.sleep(1.0)
            self.is_action_enabled = False


    def execute_turn_valve(self, valve_id, desired_increment):
        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'turn_valve'
            param_list = {'valve_id': str(valve_id),
                          'desired_increment': str(desired_increment)}
            action.parameters = __set_params__(param_list)
            while not self.is_action_enabled:
                print 'Action: \n', action
                self.pub_action_dispatch.publish(action)
                rospy.sleep(1.0)
            self.is_action_enabled = False


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
        # Init node
        rospy.init_node('test_udg_planning_interface')
        planning_interface = PlanningInterface(rospy.get_name())
        # planning_interface.execute_goto(0.0, 0.0, 2.6, -3.0)
        planning_interface.execute_check_panel()
        # planning_interface.execute_turn_valve(3, 1.57)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
