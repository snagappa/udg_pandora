#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 19 2013
@author: narcis palomeras
"""

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora_misc')
import rospy
from std_msgs.msg import Float64
from cola2_control.srv import ValveOrientation, ValveOrientationResponse

class AssistedValveTracker:
    def __init__(self, name):
        self.name = name
        self.valve_0 = 3.14
        self.valve_1 = 3.14
        self.valve_2 = 3.14
        self.valve_3 = 3.14
        
        # Create publishers
        self.pub_valve_0 = rospy.Publisher( "/valve_tracker/valve_0_ass_ori", Float64, queue_size=2)
        self.pub_valve_1 = rospy.Publisher( "/valve_tracker/valve_1_ass_ori", Float64, queue_size=2)
        self.pub_valve_2 = rospy.Publisher( "/valve_tracker/valve_2_ass_ori", Float64, queue_size=2)
        self.pub_valve_3 = rospy.Publisher( "/valve_tracker/valve_3_ass_ori", Float64, queue_size=2)
        
        # Create timer
        rospy.Timer(rospy.Duration(0.5), self.pub_valve_angles)
        
        # Create Service         
        self.change_valve_ori_srv = rospy.Service('/udg_pandora/change_valve_angle', 
                                                  ValveOrientation, 
                                                  self.change_valve_orientation)
        
        
    def pub_valve_angles(self, event):
        print 'Valves orientation: ', self.valve_0, ', ', self.valve_1, ', ', self.valve_2, ', ', self.valve_3
        self.pub_valve_0.publish( Float64(self.valve_0) )
        self.pub_valve_1.publish( Float64(self.valve_1) )
        self.pub_valve_2.publish( Float64(self.valve_2) )
        self.pub_valve_3.publish( Float64(self.valve_3) )
        
        
    def change_valve_orientation(self, req):
        print 'Change Valve ', req.valve_id, ' orientation to ', req.valve_ori
        
        if req.valve_ori < 1.57 or req.valve_ori > 3.14:
            print 'Invalid orientation'
            return ValveOrientationResponse(False)
            
        if req.valve_id == 0:
            self.valve_0 = req.valve_ori
        elif req.valve_id == 1:
            self.valve_1 = req.valve_ori
        elif req.valve_id == 2:
            self.valve_2 = req.valve_ori
        elif req.valve_id == 3:
            self.valve_3 = req.valve_ori
        else:
            print 'Invalid valve id'
            return ValveOrientationResponse(False)
        
        return ValveOrientationResponse(True)
                
    
if __name__ == '__main__':
    try:
        rospy.init_node('assisted_valve_tracker')
        AVT = AssistedValveTracker(rospy.get_name())
        rospy.spin() 
    except rospy.ROSInterruptException: 
        pass
