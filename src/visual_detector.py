#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv
from auv_msgs.msg import NavSts
from udg_pandora.msg import Detection
import metaclient
from detector import GeometricDetector
import cv2
import code
#import feature_detector
import objdetect
from cvbridge_wrapper import image_converter
import numpy as np
import image_feature_extractor

class STRUCT(object): pass

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
        
        # ROS image message to cvimage convertor
        self.ros2cvimg = image_converter()
        # Initialise panel detector
        self.panel = STRUCT()
        template_image = cv2.imread('panel_template.png', cv2.CV_8UC1)
        detector_init = False
        try:
            self.panel.detector = objdetect.detector(detector=image_feature_extractor.orb)
            detector_init = True
        except:
            print "Could not initialise ORB detector, attempting fallback to SURF"
            rospy.loginfo("Could not initialise ORB detector, attempting fallback to SURF")
            try:
                self.panel.detector = objdetect.detector(
                    detector=image_feature_extractor.surf)
                detector_init = True
            except:
                print "Failed to initialise SURF detector"
                print "Please ensure that cv2.ORB() or cv2.SURF() are available."
                rospy.loginfo("Failed to initialise SURF detector")
                rospy.logerr("Please ensure that cv2.ORB() or cv2.SURF() are available.")
        if detector_init:
            self.panel.detector.set_template(template_image)
        self.panel.DETECTED = False
        self.panel.sub = None
        # Initialise valve detector
        #self.valve = STRUCT()
        #self.valve.detector = objdetect.valve_detector()
        

    def enablePanelValveDetectionSrv(self, req):
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
        self.panel.sub = rospy.Subscriber("/uwsim/camera1", 
                                          sensor_msgs.msg.Image, 
                                          self.detect_panel)
        print "Enabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    
    def enableValveDetectionSrv(self, req):
        pass
    
    
    def enableChainDetectionSrv(self, req):
        pass
    
    
    def disablePanelValveDetectionSrv(self, req):
        if not self.panel.sub is None:
            self.panel.sub.unregister()
        print "Disabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    
    def disableValveDetectionSrv(self, req):
        pass
    
    
    def disableChainDetectionSrv(self, req):
        pass
    
    
    def updateImage(self, img):
        pass
    
    def detect_panel(self, img):
        cvimage = self.ros2cvimg.cvimagegray(img)
        self.panel.detector.detect(np.asarray(cvimage))
        self.panel.detector.show()
    
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
