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
import message_filters

class STRUCT(object): pass


class VisualDetector:
    def __init__(self, name):
        self.name = name
        self.geometric_detector = GeometricDetector('geometric_detector')
        
        # Create Subscriber
        sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
        #nav = metaclient.Subscriber("/cola2_navigation/nav_sts", NavSts, {}, self.updateNavigation)
        
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
        template_image_file = "uwsim_panel_template.png"
        rostopic_cam_root = "/stereo_camera/left"
        IS_STEREO = False
        panel_corners = np.array([[-0.8, 0.5, 0], [0.8, 0.5, 0], 
                                  [0.8, -0.5, 0], [-0.8, -0.5, 0]])
        self.panel = STRUCT()
        self.init_panel_detector( 
            template_image_file, panel_corners, rostopic_cam_root, IS_STEREO)
        
        # Initialise valve detector
        #self.valve = STRUCT()
        #self.valve.detector = objdetect.valve_detector()
        
    def init_panel_detector(self, panel_template_file, panel_corners, 
                            rostopic_camera_root="", IS_STEREO=False):
        panel = self.panel
        panel.init = False
        # Read template image
        template_image_file = (
            roslib.packages.find_resource("udg_pandora", panel_template_file))
        if len(template_image_file):
            template_image_file = template_image_file[0]
            template_image = cv2.imread(template_image_file, cv2.CV_8UC1)
        else:
            template_image = None
            rospy.logerr("Could not locate panel template")
        
        # Subscriptions
        subs = STRUCT()
        if IS_STEREO:
            # Subscribe to left and right images
            subs.camera_info = [rostopic_camera_root+"/left/camera_info",
                                rostopic_camera_root+"/right/camera_info"]
            subs.image_raw = [rostopic_camera_root+"/left/image_raw",
                              rostopic_camera_root+"/right/image_raw"]
        else:
            subs.camera_info = [rostopic_camera_root+"/camera_info"]
            subs.image_raw = [rostopic_camera_root+"/image_raw"]
        panel.subscriptions = subs
        
        # Extract camera parameters from ROS topics
        try:
            cam_info = [rospy.wait_for_message(cam_info_topic, 
                                               sensor_msgs.msg.CameraInfo, 5)
                        for cam_info_topic in subs.camera_info]
        except rospy.ROSException:
            rospy.logerr("Could not read camera parameters")
            return
        
        try:
            camera_matrix = [_cam_info_.K for _cam_info_ in cam_info]
            camera_matrix = np.reshape(np.asarray(camera_matrix), 
                                       (len(camera_matrix), 3, 3))
            camera_matrix = np.squeeze(camera_matrix)
            
            projection_matrix = [_cam_info_.P for _cam_info_ in cam_info]
            projection_matrix = np.reshape(np.asarray(projection_matrix), 
                                           (len(projection_matrix), 3, 4))
            projection_matrix = np.squeeze(projection_matrix)
        except:
            print "Error occurred"
            return
        
        _detector_ = objdetect.Detector
        func_args = (template_image, panel_corners, camera_matrix)
        if IS_STEREO:
            _detector_ = objdetect.Stereo_detector
            func_args += projection_matrix
        
        try:
            panel.detector = _detector_(*(func_args+(image_feature_extractor.orb,)))
            panel.init = True
        except AttributeError, ae:
            rospy.loginfo(ae)
            rospy.loginfo("Could not initialise ORB detector, attempting fallback to SURF")
            try:
                panel.detector = _detector_(*(func_args+(image_feature_extractor.surf,)))
                panel.init = True
            except AttributeError, ae:
                rospy.loginfo(ae)
                rospy.loginfo("Failed to initialise SURF detector")
                rospy.logerr("Please ensure that cv2.ORB() or cv2.SURF() are available.")
        panel.DETECTED = False
        panel.sub = None
        panel.img_sub = [
            message_filters.Subscriber(_sub_image_raw_, sensor_msgs.msg.Image)
            for _sub_image_raw_ in self.panel.subscriptions.image_raw]
        panel.ts = message_filters.TimeSynchronizer(panel.img_sub, 3)
        
        panel.pub = metaclient.Publisher('/visual_detector2/valve_panel', Detection,{})
        panel.detection_msg = Detection()
        
    
    def enablePanelValveDetectionSrv(self, req):
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
        self._enablePanelValveDetectionSrv_()
        print "Enabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    def _enablePanelValveDetectionSrv_(self):
        self.panel.ts.registerCallback(self.detect_panel)
        #self.panel.sub = rospy.Subscriber("/stereo_camera/left/image_raw", 
        #                                  sensor_msgs.msg.Image, 
        #                                  self.detect_panel)
        
    def enableValveDetectionSrv(self, req):
        pass
    
    
    def enableChainDetectionSrv(self, req):
        pass
    
    
    def disablePanelValveDetectionSrv(self, req):
        #if not self.panel.sub is None:
        #    self.panel.sub.unregister()
        self.panel.ts.callbacks = dict()
        print "Disabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    
    def disableValveDetectionSrv(self, req):
        pass
    
    
    def disableChainDetectionSrv(self, req):
        pass
    
    
    def updateImage(self, img):
        pass
    
    def detect_panel(self, *args):
        #self.panel.sub.unregister()
        cvimage = [np.asarray(self.ros2cvimg.cvimagegray(_img_)) for _img_ in args]
        self.panel.detector.detect(*cvimage)
        #self.panel.detector.show()
        panel_detected, panel_position = self.panel.detector.location()
        self.panel.detection_msg.detected = panel_detected
        (self.panel.detection_msg.position.position.x, 
         self.panel.detection_msg.position.position.y,
         self.panel.detection_msg.position.position.z) = panel_position
        self.panel.pub.publish(self.panel.detection_msg)
        print "Detected = ", self.panel.detection_msg.detected
        #self._enablePanelValveDetectionSrv_()
    
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
