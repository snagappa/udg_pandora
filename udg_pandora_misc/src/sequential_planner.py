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
from cola2_control.srv import StareLandmark, StareLandmarkRequest
from cola2_control.srv import ValveOrientation, ValveOrientationRequest
from std_srvs.srv import Empty, EmptyRequest
import numpy as np
from cola2_lib import cola2_lib
from pose_ekf_slam.msg import Map
import tf
import PyKDL

class PlanningInterface(object):
    def __init__(self, name):
        self.name = name
        self.map = Map()
        self.action_id = 0
        self.is_action_enabled = False
        self.is_action_running = False
        self.wait_for_valve_state = False
        self.wait_for_turn_valve = False
        self.last_valve_blocked = False
        self.wait_for_check_panel = False
        self.panel_found = False
        self.panel_in_fov = False
        self.valve_angle = [3.14, 3.14, 3.14, 3.14]

        self.inspection_points = [[-2.2, -1.5, 1.6, 3.14],
                                  [-2.2, 1.5, 1.6, 3.14],
                                  [2.2, 1.5, 1.6, 0.0],
                                  [2.2, -1.5, 1.6, 0.0]]
        self.angle_offset = 0.62

        # Create publishers
        self.pub_action_dispatch = rospy.Publisher("/planning_system/action_dispatch",
                                            ActionDispatch)

        # Create Subscribers
        rospy.Subscriber("/planning_system/action_feedback",
                         ActionFeedback,
                         self.action_feedback,
                         queue_size = 1)

        rospy.Subscriber("/pose_ekf_slam/map",
                         Map,
                         self.update_map,
                         queue_size = 1)

        # Create services to Stare landmark
        try:
            rospy.wait_for_service('/udg_pandora/change_valve_angle', 20)
            self.valve_orientation_srv = rospy.ServiceProxy(
                                '/udg_pandora/change_valve_angle', ValveOrientation)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating valve orientation client')

    def update_map(self, data):
        self.map = data

    def action_feedback(self, req):
        # rospy.loginfo('%s: Received feedback:\n %s', self.name, req)
        if req.status == "action enabled":
            self.is_action_enabled = True
        elif req.status == "action achieved":
            print 'Action achieved'
            self.is_action_running = False
            if self.wait_for_valve_state:
                for i in [0,2,4,6]:
                    self.valve_angle[i/2] = float(req.information[i].value)
                self.wait_for_valve_state = False
                print 'valves angle: ', self.valve_angle
            elif self.wait_for_check_panel:
                if len(req.information) > 1:
                    print 'panel found'
                    self.panel_found = True
                else:
                    print 'panel not found'
                    self.panel_found = False
                if req.information[0].value == "true":
                    print 'panel in FOV'
                    self.panel_in_fov = True
                else:
                    print 'panel not in FOV'
                    self.panel_in_fov = False
                self.wait_for_check_panel = False
            elif self.wait_for_turn_valve:
                self.last_valve_blocked = False
                self.wait_for_turn_valve = False
                


        elif req.status == "action failed":
            rospy.loginfo('%s: Action failed', self.name)
            self.is_action_running = False
            if self.wait_for_turn_valve:
                self.last_valve_blocked = True
                self.wait_for_turn_valve = False
                print 'Valve blocked!'


    def execute_goto(self, x, y, z, yaw):
        self.wait_to_execute('valve state')

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
            print 'Execute action goto: \n', param_list
            self.send_action(action)
            self.wait_to_execute('finish goto')


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

        self.wait_to_execute('finish valve state')

    def execute_check_panel(self):
        self.wait_to_execute('check panel')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'check_panel'
            self.wait_for_check_panel = True
            self.send_action(action)

        self.wait_to_execute('finish check panel')

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
            self.wait_to_execute('finish turn valve')


    def execute_recalibrate_arm(self):
        self.wait_to_execute('recalibrate arm')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'recalibrate_arm'

            self.send_action(action)
            self.wait_to_execute('finish recalibrate arm')


    def execute_reset_landmarks(self):
        self.wait_to_execute('reset landmarks')

        if not self.is_action_running:
            self.is_action_running = True
            action = ActionDispatch()
            action.action_id = self.action_id
            self.action_id = self.action_id + 1
            action.name = 'reset_landmarks'
            self.panel_found = False
            self.send_action(action)
            self.wait_to_execute('finish reset_landmarks')


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
        print 'wait for ', name, '...'
        while self.is_action_running and not rospy.is_shutdown():
            rospy.sleep(2.0)


    def send_action(self, action):
        while not self.is_action_enabled  and not rospy.is_shutdown():
            self.pub_action_dispatch.publish(action)
            rospy.sleep(1.0)
        self.is_action_enabled = False


    def goto_panel(self):
        in_front = False
        while not in_front:
            if self.panel_found:
                self.__go_to_pannel__(2.0)
                self.__go_to_pannel__(1.5)
                self.execute_check_panel()
                rospy.sleep(2)
                if self.panel_in_fov:
                    in_front = True
                else:
                    self.execute_reset_landmarks()
                    
            else:
                self.find_panel()
        print 'We are in front of the panel'


    def find_panel(self):
        panel_found = False
        wp = 0
        while wp < len(self.inspection_points) and not self.panel_found:
            [x, y, z, yaw] = self.rotate_wp(wp)
            print 'Go to waypoint ', wp
            self.execute_goto(x, y, z, yaw)
            self.execute_check_panel()
            rospy.sleep(2)
            wp = wp + 1
        if self.panel_found:
            print 'Panel found!'
            return True
        else:
            print 'Impossible to see the panel!'
            return False


    def turn_valve_persistently(self, valve_id, angle):
        start_time = rospy.Time.now().to_sec()

        self.goto_panel()
        self.execute_valve_state()
        rospy.sleep(2)
        turning_angle = angle - self.valve_angle[valve_id]
        print 'We are going to turn ', turning_angle, ' rads valve ', valve_id
        action_achieved = False
        valve_blocked = 0
        self.last_valve_blocked = False
        retries = 0
        while not action_achieved and valve_blocked < 2 and retries < 3 and turning_angle != 0.0:
            self.execute_turn_valve(valve_id, turning_angle)
            
            if self.last_valve_blocked:
                valve_blocked = valve_blocked + 1
            else:
                self.__set_valve_angle__(valve_id, angle)

            self.execute_recalibrate_arm()

            self.execute_valve_state()
            rospy.sleep(2)
            if angle == self.valve_angle[valve_id]:
                action_achieved = True
            else:
                turning_angle = angle - self.valve_angle[valve_id]
                print 'Retry. We are going to turn ', turning_angle, ' rads valve ', valve_id
                retries = retries + 1

        if valve_blocked >= 2:
            print 'Valve blocked'
        elif retries >= 3:
            print 'To much retries'
        else:
            print 'Valve turned correctly!'

        return rospy.Time.now().to_sec() - start_time


    def rotate_wp(self, wp):
        x = self.inspection_points[wp][0] * np.cos(self.angle_offset) - self.inspection_points[wp][1] * np.sin(self.angle_offset)
        y = self.inspection_points[wp][1] * np.cos(self.angle_offset) + self.inspection_points[wp][0] * np.sin(self.angle_offset)
        z = self.inspection_points[wp][2]
        yaw = cola2_lib.wrapAngle(self.inspection_points[wp][3] + self.angle_offset)
        return [x, y, z, yaw]


    def __set_valve_angle__(self, valve_id, valve_angle):
        print '\n\n\n--> Is valve ', valve_id, ' at ', valve_angle
        i = raw_input('--> (y/n): ')
        if i == 'y':
            print 'Setting valve ', valve_id, ' to ',  valve_angle
            req = ValveOrientationRequest()
            req.valve_id = valve_id
            req.valve_ori = valve_angle
            self.valve_orientation_srv(req)
        else:
            i = raw_input('--> enter new angle (1.57, 2.35 or 3.14): ')
            print 'Setting valve ', valve_id, ' to ',  float(i)
            req = ValveOrientationRequest()
            req.valve_id = valve_id
            req.valve_ori = float(i)
            self.valve_orientation_srv(req)
    

        
    def __go_to_pannel__(self, distance):
        rospy.sleep(2)

        # Compute offset w.r.t. panel
        Oo = PyKDL.Rotation.RPY(0.0, 1.57, 0.0)
        to = PyKDL.Vector(0.0, 0.0, distance)
        offset_transformation = PyKDL.Frame(Oo, to)

        # Compute panel pose w.r.t. world
        angle = tf.transformations.euler_from_quaternion(
                    [self.map.landmark[0].pose.pose.orientation.x,
                     self.map.landmark[0].pose.pose.orientation.y,
                     self.map.landmark[0].pose.pose.orientation.z,
                     self.map.landmark[0].pose.pose.orientation.w])
        O = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
        t = PyKDL.Vector(self.map.landmark[0].pose.pose.position.x,
                         self.map.landmark[0].pose.pose.position.y,
                         self.map.landmark[0].pose.pose.position.z)
        wTl = PyKDL.Frame(O, t)

        # Compose offset w.r.t. panel with panel pose w.r.t. world
        wTo = wTl * offset_transformation
        orientation = wTo.M.GetRPY()
        position = wTo.p

        # Go to
        self.execute_goto(position[0], position[1], position[2], orientation[2])


def __set_params__(param_list):
    ret = list()
    for key in param_list:
        element = KeyValue()
        element.key = key
        element.value = param_list[key]
        ret.append(element)
    return ret


def __sleep__(elapsed_time, desired_time):
    t = desired_time - elapsed_time
    if t < 0.0:
        t = 0.0
    print 'Wait ', t, ' seconds for next goal'
    rospy.sleep(t)


if __name__ == '__main__':
    try:
        # Init node
        rospy.init_node('test_udg_planning_interface')
        planning_interface = PlanningInterface(rospy.get_name())
        small_timeout = 20
        large_timeout = 50
        
        while not rospy.is_shutdown():
            t = planning_interface.turn_valve_persistently(0, 1.57)
            #__sleep__(t, large_timeout)
            t = planning_interface.turn_valve_persistently(1, 1.57)
            #__sleep__(t, small_timeout)
            #t = planning_interface.turn_valve_persistently(2, 1.57)
            #__sleep__(t, small_timeout)
            t = planning_interface.turn_valve_persistently(3, 1.57)
            #__sleep__(t, small_timeout)
            #print 'CHANGE PANEL POSITION!!! You have 5 minuts to do it.'
            #rospy.sleep(300)
            t = planning_interface.turn_valve_persistently(0, 3.14)
            #__sleep__(t, large_timeout)
            t = planning_interface.turn_valve_persistently(1, 3.14)
            #__sleep__(t, small_timeout)
            #t = planning_interface.turn_valve_persistently(2, 3.14)
            #__sleep__(t, small_timeout)
            t = planning_interface.turn_valve_persistently(3, 3.14)
            #__sleep__(t, small_timeout)
            #print 'CHANGE PANEL POSITION!!! You have 5 minuts to do it.'
            #rospy.sleep(300)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
