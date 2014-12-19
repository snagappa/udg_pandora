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
        self.wait_for_valve_state = False
        self.wait_for_turn_valve = False
        self.last_valve_blocked = False

        self.valve_angle = [3.14, 3.14, 3.14, 3.14]

        # Create publishers
        self.pub_action_dispatch = rospy.Publisher("/planning_system/action_dispatch",
                                            ActionDispatch)

        # Create Subscribers
        rospy.Subscriber("/planning_system/action_feedback",
                         ActionFeedback,
                         self.action_feedback,
                         queue_size = 1)


    def action_feedback(self, req):
        rospy.loginfo('%s: Received feedback:\n %s', self.name, req)
        if req.status == "action enabled":
            self.is_action_enabled = True
        elif req.status == "action achieved":
            self.is_action_running = False
            if self.wait_for_valve_state:
                for i in [0,2,4,6]:
                    self.valve_angle[i/2] = float(req.information[i].value)
                self.wait_for_valve_state = False
                rospy.loginfo('%s: valves angle:\n %s', self.name, self.valve_angle)
        elif req.status == "action failed":
            rospy.loginfo('%s: Action failed', self.name)
            self.is_action_running = False
            if self.wait_for_turn_valve:
                self.last_valve_blocked = True
                self.wait_for_turn_valve = False
                rospy.loginfo('%s: Valve blocked', self.name)




    def execute_goto(self, x, y, z, yaw):
	while self.is_action_running:
            rospy.sleep(1.0)            
            print 'wait for goto...'

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
            self.send_action(action)


    def execute_valve_state(self):
        self.wait_to_execute('valve state')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'valve_state'
            self.wait_for_valve_state = True
            self.send_action(action)


    def execute_check_panel(self):
        self.wait_to_execute('check panel')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'check_panel'

            self.send_action(action)


    def execute_turn_valve(self, valve_id, desired_increment):
        self.wait_to_execute('turn valve')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            param_list = {'valve_id': str(valve_id),
                          'desired_increment': str(desired_increment)}
            action.name = 'turn_valve'
            action.parameters = __set_params__(param_list)
            self.wait_for_turn_valve = True

            self.send_action(action)


    def execute_recalibrate_arm(self):
        self.wait_to_execute('recalibrate arm')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'recalibrate_arm'
            
            self.send_action(action)

    def execute_chain_follow(self):
        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'enable_chain_follow'
            while not self.is_action_enabled:
                print 'Action: \n', action
                self.pub_action_dispatch.publish(action)
                rospy.sleep(1.0)
            self.is_action_enabled = False

            rospy.sleep(10.0)
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'disable_chain_follow'

            self.send_action(action)



    def wait_to_execute(self, name):
        while self.is_action_running and not rospy.is_shutdown():
            rospy.sleep(1.0)
            print 'wait for ', name, '...'


    def send_action(self, action):
        while not self.is_action_enabled  and not rospy.is_shutdown():
            self.pub_action_dispatch.publish(action)
            rospy.sleep(1.0)
        self.is_action_enabled = False

    def turn_valve_persistently(self, valve_id, angle):
        planning_interface.execute_valve_state()
        rospy.sleep(10)
        turning_angle = angle - self.valve_angle[valve_id]
        print 'We are going to turn ', turning_angle, ' rads'
        action_achieved = False
        valve_blocked = 0
        retries = 0
        while not action_achieved and valve_blocked < 2 and retries < 3:
            planning_interface.execute_turn_valve(valve_id, turning_angle)
            planning_interface.execute_recalibrate_arm()

            if self.last_valve_blocked:
                valve_blocked = valve_blocked + 1

            planning_interface.execute_valve_state()
            rospy.sleep(10)
            if angle == self.valve_angle[valve_id]:
                action_achieved = True
            else:
                turning_angle = angle - self.valve_angle[valve_id]
                print 'Retry. We are going to turn ', turning_angle, ' rads'
                retries = retries + 1

        if valve_blocked >= 2:
            print 'Valve blocked'
        elif retries >= 3:
            print 'To much retries'
        else:
            print 'Valve turned correctly!'


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
        planning_interface.turn_valve_persistently(2, 1.57)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
