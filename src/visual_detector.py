#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import std_msgs.msg
import std_srvs.srv
from auv_msgs.msg import NavSts
from udg_pandora.msg import Detection
import metaclient
from detector import GeometricDetector

class VisualDetector:
    def __init__(self, name):
        self.name = name
        self.geometric_detector = GeometricDetector('geometric_detector')
        
        # Create Subscriber
        sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
        nav = metaclient.Subscriber("/cola2_navigation/nav_sts", NavSts, {}, self.updateNavigation)
        
        # Create publisher
        self.pub_valve_panel = metaclient.Publisher('/visual_detector/valve_panel', Detection,{})
        self.pub_valve = metaclient.Publisher('/visual_detector/valve', Detection, {})
        self.pub_chain = metaclient.Publisher('/visual_detector/chain', Detection, {})
        
        #Create services
        self.enable_panel_valve_detection = metaclient.Service('/visual_detector/enable_panel_valve_detection', std_srvs.srv.Empty, self.enablePanelValveDetectionSrv, {})
        self.enable_valve_detection = metaclient.Service('/visual_detector/enable_valve_detection', std_srvs.srv.Empty, self.enableValveDetectionSrv, {})
        self.enable_chain_detection = metaclient.Service('/visual_detector/enable_chain_detection', std_srvs.srv.Empty, self.enableChainDetectionSrv, {})
        self.disable_panel_valve_detection = metaclient.Service('/visual_detector/disable_panel_valve_detection', std_srvs.srv.Empty, self.disablePanelValveDetectionSrv, {})
        self.disable_valve_detection = metaclient.Service('/visual_detector/disable_valve_detection', std_srvs.srv.Empty, self.disableValveDetectionSrv, {})
        self.disable_chain_detection = metaclient.Service('/visual_detector/disable_chain_detection', std_srvs.srv.Empty, self.disableChainDetectionSrv, {})


    def enablePanelValveDetectionSrv(self, req):
        pass
    
    
    def enableValveDetectionSrv(self, req):
        pass
    
    
    def enableChainDetectionSrv(self, req):
        pass
    
    
    def disablePanelValveDetectionSrv(self, req):
        pass
    
    
    def disableValveDetectionSrv(self, req):
        pass
    
    
    def disableChainDetectionSrv(self, req):
        pass
    
    
    def updateImage(self, img):
        pass
    
    def updateNavigation(self, nav):
        vehicle_pose = [nav.position.north, nav.position.east, nav.position.depth]
        vehicle_orientation = [nav.orientation.roll, nav.orientation.pitch, nav.orientation.yaw]
        d = self.geometric_detector.detectObjects(vehicle_pose, vehicle_orientation)
        for i in range(len(d)):
            if d[i].detected:       
                if i == 0:
                    self.pub_valve_panel.publish(d[i])
                elif i == 1:
                    self.pub_valve.publish(d[i])
                else:
                    self.pub_chain.publish(d[i])
    
if __name__ == '__main__':
    try:
        rospy.init_node('visual_detector')
        visual_detector = VisualDetector(rospy.get_name())
        rospy.spin() 
    except rospy.ROSInterruptException: pass
