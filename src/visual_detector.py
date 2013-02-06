#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import CameraInfo, Image
#import std_msgs.msg
import std_srvs.srv
#from auv_msgs.msg import NavSts
from udg_pandora.msg import Detection
import hwu_meta_data.metaclient as metaclient
from detector import GeometricDetector
import cv2
import code
from lib import objdetect
import numpy as np
import message_filters
from lib.misctools import STRUCT, image_converter, Retinex
from lib import image_feature_extractor
from tf import TransformBroadcaster, TransformListener
from tf.transformations import quaternion_from_euler, euler_from_quaternion, \
    quaternion_matrix
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading
import copy
import pickle
import itertools
#from lib.cameramodels import transform_numpy_array
#import time

USE_SIMULATOR = False
if USE_SIMULATOR:
    ROSTOPIC_CAM_ROOT = "/stereo_front"
else:
    ROSTOPIC_CAM_ROOT = "/stereo_camera"
ROSTOPIC_CAM_SUB = "image_rect"


# Check if ORB/SURF detector is available
try:
    _default_feature_extractor_ = image_feature_extractor.Orb().__class__
except AttributeError as attr_err:
    print attr_err
    try:
        _default_feature_extractor_ = image_feature_extractor.Surf().__class__
    except AttributeError as attr_err:
        print attr_err
        rospy.logfatal("No feature extractors available!")
        raise rospy.exceptions.ROSException(
            "Cannot initialise feature extractors")


class stereo_image_buffer(object):
    def __init__(self, rostopic_camera_root="", image_sub_topic="/image_rect"):
        self._image_messages_ = [None, None]
        #self._cv_images_ = [None, None]
        self._message_time_ = rospy.Time(0)
        # Convert between sensor_msgs Image anc CV image
        self.ros2cvimg = image_converter()

        # Initialise topics as None
        self._camera_info_topic_ = [None, None]
        self._image_topic_ = [None, None]
        self._camera_info_ = (None, None)
        
        # Create the topic names we need to subscribe to
        if ((type(rostopic_camera_root) is str) and
            (type(image_sub_topic) is str)):
            # Strip trailing '/'
            rostopic_camera_root = rostopic_camera_root.rstrip("/")
            # Strip leading '/'
            image_sub_topic = image_sub_topic.lstrip("/")
            self._camera_info_topic_ = (
                [rostopic_camera_root+"/left/camera_info",
                 rostopic_camera_root+"/right/camera_info"])
            self._image_topic_ = (
                [rostopic_camera_root+"/left/"+image_sub_topic,
                 rostopic_camera_root+"/right/"+image_sub_topic])
        
        # Try to subscribe to the camera info
        self._cam_info_from_topics_()
        
        # List of callbacks and rates
        self._callbacks_ = {}
        self._last_callback_id_ = -1
        self._last_callback_time_ = []
        self._lock_ = threading.Lock()
        
        # Subscribe to left and right images
        self._subscribe_()
        
    def _cam_info_from_topics_(self):
        self._camera_info_ = [None, None]
        try:
            self._camera_info_ = tuple(
                [rospy.wait_for_message(cam_info_topic, CameraInfo, 2)
                 for cam_info_topic in self._camera_info_topic_])
        except rospy.ROSException:
            rospy.logerr("Could not read camera parameters")
            camera_pickle_file = "bumblebee.p"
            print "Loading information from "+camera_pickle_file
            camera_info_pickle = roslib.packages.find_resource("udg_pandora",
                camera_pickle_file)
            if len(camera_info_pickle):
                camera_info_pickle = camera_info_pickle[0]
                try:
                    self._camera_info_ = tuple(
                        pickle.load(open(camera_info_pickle, "rb")))
                except IOError:
                    print "Failed to load camera information!"
                    rospy.logfatal("Could not read camera parameters")
                    raise rospy.exceptions.ROSException(
                        "Could not read camera parameters")
            else:
                print "Failed to load camera information!"
                rospy.logfatal("Could not read camera parameters")
                raise rospy.exceptions.ROSException(
                    "Could not read camera parameters")
    
    def fromCameraInfo(self, camera_info_left, camera_info_right=None):
        self._camera_info_ = tuple([copy.deepcopy(_info_)
            for _info_ in (camera_info_left, camera_info_right)])
    
    def _subscribe_(self):
        self._img_sub_ = [
            message_filters.Subscriber(_sub_image_raw_, Image)
            for _sub_image_raw_ in self._image_topic_]
        self.timesync = message_filters.TimeSynchronizer(self._img_sub_, 10)
        self.timesync.registerCallback(self.update_images)
    
    def update_images(self, *args):
        self._lock_.acquire()
        if len(self._callbacks_):
            self._image_messages_ = [_img_ for _img_ in args]
            #self._cv_images_ = (
            #    [np.asarray(self.ros2cvimg.cvimagegray(_img_)) for _img_ in args])
            self._message_time_ = self._image_messages_[0].header.stamp
            self._start_callback_threads_()
        self._lock_.release()
    
    def register_callback(self, callback, rate=None):
        self._lock_.acquire()
        self._last_callback_id_ += 1
        time_delay = None if rate is None else rospy.Duration(1./rate)
        self._callbacks_[self._last_callback_id_] = {"callback":callback,
                                                     "timedelay":time_delay,
                                                     "lasttime":rospy.Time(0),
                                                     "block":0}
        self._lock_.release()
        return self._last_callback_id_
    
    def unregister_callback(self, callback_id):
        self._lock_.acquire()
        try:
            self._callbacks_.pop(callback_id)
        except KeyError:
            print "callback_id not found. already unregistered?"
        finally:
            self._lock_.release()
    
    def _start_callback_threads_(self):
        #self._lock_.acquire()
        threads = []
        unregister_callbacks_list = []
        for _callback_ in self._callbacks_:
            try:
                callback_info = self._callbacks_[_callback_]
                if ((not callback_info["block"]) and
                    (callback_info["timedelay"] is None or
                    self._message_time_ - callback_info["lasttime"] >= callback_info["timedelay"])):
                    threads.append(threading.Thread(target=self._thread_wrapper_, args=(callback_info,)))
                    threads[-1].start()
                    callback_info["lasttime"] = copy.copy(self._message_time_)
            except:
                print "error in callback, unregistering..."
                unregister_callbacks_list.append(_callback_)
        for _callback_ in unregister_callbacks_list:
            self._callbacks_.pop(_callback_)
        #self._lock_.release()
    
    def _thread_wrapper_(self, callback_info):
        callback_info["block"] = 1
        callback_info["callback"](*self._image_messages_)
        callback_info["block"] = 0
    
    def get_camera_info(self, idx=None):
        if not idx is None:
            return (self._camera_info_[idx],)
        else:
            return self._camera_info_
    

class VisualDetector(object):
    def __init__(self, name, rostopic_cam_root=ROSTOPIC_CAM_ROOT, 
                 image_sub_topic=ROSTOPIC_CAM_SUB):
        self.name = name
        self.rostopic_cam_root = rostopic_cam_root
        self.image_sub_topic = image_sub_topic
        
        self.publish_transforms()
        rospy.timer.Timer(rospy.Duration(0.1), self.publish_transforms)
        self.geometric_detector = GeometricDetector('geometric_detector')
        
        # Create Subscriber
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty,
        #                            {}, self.updateImage)
        #nav = metaclient.Subscriber("/cola2_navigation/nav_sts", NavSts, {},
        #                            self.updateNavigation)
        
        # Create publisher
        self.pub_valve_panel = metaclient.Publisher(
            '/visual_detector/valve_panel', Detection,{})
        self.pub_valve = metaclient.Publisher('/visual_detector/valve',
                                              Detection, {})
        self.pub_chain = metaclient.Publisher('/visual_detector/chain',
                                              Detection, {})
        
        # Valve detection
        self.valve = STRUCT()
        try:
            self.valve.tflistener = TransformListener()
        except:
            rospy.logerr("VISUALDETECTOR::INIT(): Error occurred while initialising tflistener")
        #self.valve.sub = metaclient.Subscriber(
            #"/visual_detector2/valve_panel", Detection, {}, self.detect_valves)
        self.valve.pub = STRUCT()
        self.valve.pub.img = metaclient.Publisher(
            "/visual_detector2/valve_img", Image, {})
        for valve_count in range(6):
            setattr(self.valve.pub, "v"+str(valve_count),
                    metaclient.Publisher('/visual_detector2/valve'+str(valve_count), 
                                         Detection, {}))
        
        #Create services
        self.enable_panel_valve_detection = metaclient.Service('/visual_detector/enable_panel_valve_detection', std_srvs.srv.Empty, self.enablePanelValveDetectionSrv, {})
        self.enable_valve_detection = metaclient.Service('/visual_detector/enable_valve_detection', std_srvs.srv.Empty, self.enableValveDetectionSrv, {})
        self.enable_chain_detection = metaclient.Service('/visual_detector/enable_chain_detection', std_srvs.srv.Empty, self.enableChainDetectionSrv, {})
        self.disable_panel_valve_detection = metaclient.Service('/visual_detector/disable_panel_valve_detection', std_srvs.srv.Empty, self.disablePanelValveDetectionSrv, {})
        self.disable_valve_detection = metaclient.Service('/visual_detector/disable_valve_detection', std_srvs.srv.Empty, self.disableValveDetectionSrv, {})
        self.disable_chain_detection = metaclient.Service('/visual_detector/disable_chain_detection', std_srvs.srv.Empty, self.disableChainDetectionSrv, {})
        
        # ROS image message to cvimage convertor
        self.ros2cvimg = image_converter()
        
        # Initialise image buffer
        self.image_buffer = stereo_image_buffer(self.rostopic_cam_root,
                                                self.image_sub_topic)
        
        ######################### Panel Templates #########################
        # Initialise panel detector
        if not USE_SIMULATOR:
            # For new real panel:
            template_image_file = ["new_panel_template2.png"]
            template_mask_file = [] #["new_panel_mask.png"]
        else:
            # For simulator:
            template_image_file = ["simulator_panel.png"]
            template_mask_file = []
        
        ######################### Panel Templates #########################
        
        panel_corners = np.array([[-0.8, 0.5, 0], [0.8, 0.5, 0],
                                  [0.8, -0.5, 0], [-0.8, -0.5, 0]])/2
        panel_update_rate = None
        self.panel = STRUCT()
        self.tf_broadcaster = TransformBroadcaster()
        IS_STEREO = False
        self.init_panel_detector(template_image_file, panel_corners,
            template_mask_file, IS_STEREO, panel_update_rate)
        # Enable panel detection by default
        self._enablePanelValveDetectionSrv_()
        print "Completed initialisation"
    
    def publish_transforms(self, *args, **kwargs):
        timestamp = rospy.Time.now()
        o_zero = quaternion_from_euler(0, 0, 0, 'sxyz')
        # Publish stereo_camera tf
        camera_tf_br = TransformBroadcaster()
        o_zero = quaternion_from_euler(0, 0, 0, 'sxyz')
        o_st_down = quaternion_from_euler(0.0, 0.0, 1.57, 'sxyz')
        t_st_down = (0.6, 0.0, 0.4)
        o_st_front = quaternion_from_euler(1.57, 0.0, 1.57, 'sxyz')
        t_st_front = (0.6, 0.0, 0.2)
        t_baseline = (0.12, 0.0, 0.0)
        # Stereo down
        camera_tf_br.sendTransform(t_st_down, o_st_down, timestamp,
                                   '/stereo_down', 'girona500')
        camera_tf_br.sendTransform(t_baseline, o_zero, timestamp,
                                   '/stereo_down_right', '/stereo_down')
        
        # Bumblebee tf
        camera_tf_br.sendTransform(t_st_front, o_st_front, timestamp,
                                   'bumblebee2', 'girona500')
        camera_tf_br.sendTransform(t_baseline, o_zero, timestamp,
                                   'bumblebee2_right', 'bumblebee2')
    
    def init_panel_detector(self, template_file_list, panel_corners,
                            template_mask_file_list=[], IS_STEREO=False,
                            panel_update_rate=None):
        panel = self.panel
        panel.init = False
        
        if IS_STEREO:
            panel.detector = objdetect.Stereo_detector(
                feat_detector=_default_feature_extractor_)
        else:
            panel.detector = objdetect.Detector(
                feat_detector=_default_feature_extractor_)
        # Set number of features from detector
        panel.detector.set_detector_num_features(4000)
        try:
            panel.detector.camera._featuredetector_.set_scaleFactor(1.06)
            panel.detector.camera._featuredetector_.set_nLevels(16)
        except:
            print "Error setting scaleFactor and nLevels"
        
        # Plugin for white balance
        panel.Retinex = Retinex()
        
        # Get camera info msg and initialise camera
        cam_info = self.image_buffer.get_camera_info()
        panel.detector.init_camera(*cam_info)
        panel.detector.camera.set_tf_frame(self.rostopic_cam_root,
                                           self.rostopic_cam_root+"_right")
        # Set template
        panel.detector.set_corners3d(panel_corners)
        # Read template image
        for panel_template_file, panel_template_mask_file in (
            itertools.izip_longest(template_file_list, template_mask_file_list)):
            template_image_file = (
                roslib.packages.find_resource("udg_pandora", panel_template_file))
            if len(template_image_file):
                template_image_file = template_image_file[0]
                template_image = cv2.imread(template_image_file, cv2.CV_8UC1)
            else:
                rospy.logerr("Could not locate panel template")
                raise rospy.exceptions.ROSException(
                        "Could not locate panel template")
            
            # Read template mask
            template_mask = None
            if not template_mask_file_list is None:
                template_mask_file = (roslib.packages.find_resource(
                    "udg_pandora", panel_template_mask_file))
            if len(template_mask_file):
                template_mask_file = template_mask_file[0]
                template_mask = cv2.imread(template_mask_file, cv2.CV_8UC1)
            
            # Fix luminance before adding template
            #try:
            #    template_image2 = panel.Retinex.retinex(template_image).astype(template_image.dtype)
            #    panel.detector.add_to_template(template_image2, template_mask)
            #except:
            #    print "Error using retinex()"
            panel.detector.add_to_template(template_image, template_mask)
        
        self.panel.callback_id = None
        self.panel.update_rate = panel_update_rate
        
        # Publish panel position
        panel.pub = metaclient.Publisher('/visual_detector2/valve_panel',
                                         Detection, {})
        panel.detection_msg = Detection()
        panel.pose_msg = PoseWithCovarianceStamped()
        panel.pose_msg.header.frame_id = cam_info[0].header.frame_id #"panel_centre"
        panel.pose_msg_pub = metaclient.Publisher(
            '/pose_ekf_slam/landmark_update/panel_centre',
            PoseWithCovarianceStamped, {})
        # Publish image of detected panel
        self.panel.img_pub = metaclient.Publisher(
            '/visual_detector2/panel_img', Image, {})
    
    def enablePanelValveDetectionSrv(self, req):
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty,
        #                            {}, self.updateImage)
        self._enablePanelValveDetectionSrv_()
        print "Enabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    def _enablePanelValveDetectionSrv_(self):
        #self.panel.ts.registerCallback(self.detect_panel)
        self.panel.callback_id = (
            self.image_buffer.register_callback(self.detect_panel,
                                                self.panel.update_rate))
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
        #self.panel.ts.callbacks = dict()
        self.image_buffer.unregister_callback(self.panel.callback_id)
        print "Disabled panel detection"
        return std_srvs.srv.EmptyResponse()
    
    def disableValveDetectionSrv(self, req):
        pass
    
    def disableChainDetectionSrv(self, req):
        pass
    
    def detect_panel(self, *args):
        #self.panel.sub.unregister()
        _cvimage_ = [np.asarray(self.ros2cvimg.cvimagegray(_img_)).copy()
            for _img_ in args]
        #try:
        #    cvimage = [self.panel.Retinex.retinex(_img_).astype(_img_.dtype)
        #        for _img_ in _cvimage_]
        #except:
        #    print "Error using retinex()"
        cvimage = _cvimage_
        self.panel.detector.detect(*cvimage)
        #self.panel.detector.show()
        panel_detected, panel_centre, panel_orientation = (
            self.panel.detector.location())
        panel_rpy = panel_orientation#[[2, 0, 1]]
        valid_panel_orientation = self._check_valid_panel_orientation_(panel_rpy)
        if valid_panel_orientation:
            bbox_colour = (255, 255, 255)
        else:
            bbox_colour = (0, 0, 0)
        if panel_detected:
            corners = self.panel.detector.obj_corners.astype(np.int32)
            out_img = self.panel.detector.get_scene(0).copy()
            cv2.polylines(out_img, [corners],
                          True, bbox_colour, 6)
        else:
            out_img = self.panel.detector.get_scene(0)
        time_now = rospy.Time.now()
        self.panel.detection_msg.header.stamp = time_now
        panel_detected = (panel_detected and valid_panel_orientation)        
        panel_orientation_quaternion = quaternion_from_euler(*panel_rpy)
        self.panel.detection_msg.detected = panel_detected
        (self.panel.detection_msg.position.position.x,
         self.panel.detection_msg.position.position.y,
         self.panel.detection_msg.position.position.z) = panel_centre
        (self.panel.detection_msg.position.orientation.x,
         self.panel.detection_msg.position.orientation.y,
         self.panel.detection_msg.position.orientation.z,
         self.panel.detection_msg.position.orientation.w) = (
             panel_orientation_quaternion)
        self.panel.pub.publish(self.panel.detection_msg)
        
        if panel_detected:
            # Panel orientation is screwy - only broadcast the yaw
            # which is panel_orientation[1]
            self.panel.pose_msg.header.stamp = time_now
            (self.panel.pose_msg.pose.pose.position.x,
             self.panel.pose_msg.pose.pose.position.y,
             self.panel.pose_msg.pose.pose.position.z) = panel_centre
            (self.panel.pose_msg.pose.pose.orientation.x,
             self.panel.pose_msg.pose.pose.orientation.y,
             self.panel.pose_msg.pose.pose.orientation.z,
             self.panel.pose_msg.pose.pose.orientation.w) = (
                 panel_orientation_quaternion)
            cov = np.zeros((6, 6))
            pos_diag_idx = range(3)
            cov[pos_diag_idx, pos_diag_idx] = (
                (((1.2*np.linalg.norm(panel_centre))**2)*0.03)**2)
            self.panel.pose_msg.pose.covariance = cov.flatten().tolist()
            self.panel.pose_msg_pub.publish(self.panel.pose_msg)
            self.tf_broadcaster.sendTransform(tuple(panel_centre),
                panel_orientation_quaternion,
                time_now, "/panel_centre", "bumblebee2")
            # Detect valves if panel was detected at less than 2 metres
            self.detect_valves(self.panel.pose_msg)
            #print "\nPanel RPY = %s \n" % (panel_rpy*180/np.pi)
            
        # Publish image of detected panel
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(out_img))
        img_msg.header.stamp = time_now
        self.panel.img_pub.publish(img_msg)
        return
        #print "Detected = ", self.panel.detection_msg.detected
        #self._enablePanelValveDetectionSrv_()
        """
        Detect panel using contours
        # Filter the image by downscaling and upscaling
        src_img = cvimage[0].copy()
        filt_img = cv2.pyrDown(src_img, dstsize=(src_img.shape[1]/2, src_img.shape[0]/2))
        filt_img = cv2.pyrUp(filt_img, dstsize=(src_img.shape[1], src_img.shape[0]))
        #filt_img = src_img
        # Edge detection
        edges_img = cv2.Canny(filt_img, 0, 100, apertureSize=5, L2gradient=True)
        # Dilate the image
        dilate_img = cv2.dilate(edges_img, np.ones((5, 5)))
        # Extract contours
        contours, hierarchy = cv2.findContours(dilate_img, mode=cv2.RETR_TREE, 
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        def _angle_(pt1, pt2, pt0):
            dx1 = pt1[0] - pt0[0]
            dy1 = pt1[1] - pt0[1]
            dx2 = pt2[0] - pt0[0]
            dy2 = pt2[1] - pt0[1]
            return (dx1*dx2 + dy1*dy2)/np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10)
        detected_rects = []
        
        for idx in range(len(contours)):
            contour_length = cv2.arcLength(contours[idx], True)
            approx = cv2.approxPolyDP(contours[idx], contour_length*0.02, True)
            if ((np.squeeze(approx).shape[0] == 4) and
                    (np.fabs(cv2.contourArea(approx)) > 5000) and 
                    cv2.isContourConvex(approx)):
                maxCosine = 0
                for j in range(2, 5):
                    cosine = np.fabs(_angle_(approx[j%4][0], approx[j-2][0], approx[j-1][0]))
                    maxCosine = max((maxCosine, cosine))
                if( maxCosine < 0.3 ):
                    detected_rects.append(approx)
        
        out_img = self.panel.detector.get_scene(0).copy()
        color = (255, 255, 255)
        cv2.drawContours(out_img, detected_rects, -1, color, 3)
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(edges_img))
        img_msg.header.stamp = time_now
        self.panel.img_pub.publish(img_msg)
        #code.interact(local=locals())
        return
        """
    
    def _check_valid_panel_orientation_(self, panel_rpy):
        # if panel_orientation{0,2} not close to zero, return false
        abs_rpy = np.abs(panel_rpy)
        if (((abs_rpy[0] < 0.35) or ((np.pi-abs_rpy[0]) < 0.35))):
            #and ((abs_rpy[1] < 0.35) or ((np.pi-abs_rpy[1]) < 0.35))):
            return True
        else:
            return False
        #remainder = np.mod(np.pi - np.mod(abs_po, np.pi), np.pi)
        #if np.any(remainder > 0.2):
        #    return False
        #else:
        #    return True
        #return True
    
    def detect_valves(self, detection_msg):
        if type(detection_msg) is Detection:
            panel_centre = np.array((detection_msg.position.position.x,
                                     detection_msg.position.position.y,
                                     detection_msg.position.position.z))
            panel_orientation_quaternion = np.asarray(
                (detection_msg.position.orientation.x,
                 detection_msg.position.orientation.y,
                 detection_msg.position.orientation.z,
                 detection_msg.position.orientation.w))
            panel_detected = detection_msg.detected
        elif type(detection_msg) is PoseWithCovarianceStamped:
            panel_detected = True
            panel_centre = np.asarray(
                (detection_msg.pose.pose.position.x,
                 detection_msg.pose.pose.position.y,
                 detection_msg.pose.pose.position.z))
            panel_orientation_quaternion = np.asarray(
                (detection_msg.pose.pose.orientation.x,
                 detection_msg.pose.pose.orientation.y,
                 detection_msg.pose.pose.orientation.z,
                 detection_msg.pose.pose.orientation.w))
            panel_orientation_euler = np.asarray(
                euler_from_quaternion(panel_orientation_quaternion))
        # Get location of valves if closer than ~2m and orientation < ~30 degrees
        # Panel positions (in m) are (relative to 0,0 centre):
        #   left: (-0.5, +-0.25), centre: (0, +-0.25), right: (0.5, +-0.25)
        # Stem length is 10-11cm, t-bar length is 10cm
        scene = self.panel.detector.get_scene(0).copy()
        # Equalise the histogram
        #scene = cv2.equalizeHist(scene)
        # Get transformation of panel centre from
        # (x=0, y=0, z=0, r=0, p=0, y=0), (panel_centre, panel_orientation)
        if (panel_detected and (np.linalg.norm(panel_centre) <= 1.5) and 
            (np.abs(panel_orientation_euler[1]) < 0.5)):
            # Wait until transform becomes available
            homogenous_rotation_matrix = (
                quaternion_matrix(panel_orientation_quaternion))
            homogenous_rotation_matrix[:3, -1] = panel_centre
            """
            try:
                self.valve.tflistener.waitForTransform("bumblebee2", 
                    "/panel_centre", rospy.Time(), rospy.Duration(0.5))
            except tf.Exception:
                rospy.logerr("DETECT_VALVES(): Could not get transform from panel_centre to bumblebee2")
                return
            """
            time_now = detection_msg.header.stamp #rospy.Time.now()
            valve_z = 0.11
            # Valve centres in homogenous coordinates
            valve_centre = np.array([[-0.25, -0.125, valve_z, 1],
                                     #[-0.25, +0.125, valve_z, 1],
                                     [ 0.0, -0.125, valve_z, 1],
                                     [ 0.0, +0.125, valve_z, 1],
                                     [+0.25, -0.125, valve_z, 1],
                                     #[+0.25, +0.125, valve_z, 1]
                                     ])
            #valve_centre[:, :2] += np.squeeze(panel_centre)[:2]
            valve_rad = 0.05
            # Bounding box for each valve
            valve_bbox = np.array([[-valve_rad, -valve_rad, 0, 0],
                                   [ valve_rad, -valve_rad, 0, 0],
                                   [ valve_rad,  valve_rad, 0, 0],
                                   [-valve_rad,  valve_rad, 0, 0]])
            valve_msg = Detection()
            valve_msg.header.stamp = time_now
            valve_msg.header.frame_id = "panel_centre"
            for valve_idx in range(valve_centre.shape[0]):
                # Get bounding box for the current valve
                this_valve = valve_centre[valve_idx] + valve_bbox
                # Convert points from panel to camera co-ordinates
                #this_valve_camcoords = transform_numpy_array("bumblebee2", "panel_centre", this_valve)
                this_valve_camcoords = np.dot(homogenous_rotation_matrix, this_valve.T).T
                # Project the 3D points from camera coordinates to pixels
                px_corners = self.panel.detector.camera.project3dToPixel(this_valve_camcoords[:, :3])
                px_corners = px_corners.astype(np.int32)
                left, top = np.min(px_corners, axis=0)
                right, bottom = np.max(px_corners, axis=0)
                if left < 0 or top < 0 or right > scene.shape[1] or bottom > scene.shape[0]:
                    pass
                else:
                    # Get approximate length of the valve in pixels
                    bbox_side_lengths = ((np.diff(np.vstack((px_corners, px_corners[0])), axis=0)**2).sum(axis=1))**0.5
                    valve_length_px = np.int32(0.8*np.mean(bbox_side_lengths))
                    valve_im_bbox = (left, top, right, bottom)
                    this_valve_orientation = self.detect_valve_orientation(
                        scene, valve_im_bbox, HIGHLIGHT=True,
                        HoughMinLineLength=valve_length_px)
                    if not this_valve_orientation is None:
                        valve_msg.detected = True
                        (valve_msg.position.position.x,
                         valve_msg.position.position.y,
                         valve_msg.position.position.z) = valve_centre[valve_idx][:3]
                        # Obtain rotation of the valve wrt the panel
                        # TODO: Find a more robust way of doing this
                        panel_rpy = panel_orientation_euler
                        valve_rpy = (0, 0, this_valve_orientation - panel_rpy[2])
                        valve_orientation_quaternion = quaternion_from_euler(*valve_rpy)
                        (valve_msg.position.orientation.x,
                         valve_msg.position.orientation.y,
                         valve_msg.position.orientation.z,
                         valve_msg.position.orientation.w) = valve_orientation_quaternion
                        v_pub = getattr(self.valve.pub, "v"+str(valve_idx))
                        v_pub.publish(valve_msg)
                        self.tf_broadcaster.sendTransform(tuple(valve_centre[valve_idx]),
                            valve_orientation_quaternion, time_now,
                            "valve"+str(valve_idx), "panel_centre")
                        #valve_centre_3d = transform_numpy_array("bumblebee2", "panel_centre", valve_centre[np.newaxis, valve_idx])
                        #v_centre_px = self.panel.detector.camera.project3dToPixel(valve_centre_3d)[0]
                        #try:
                        #    scene[v_centre_px[1], v_centre_px[0]] = 255
                        #    scene[px_corners[:, 1], px_corners[:, 0]] = 255
                        #except IndexError:
                        #    pass
                cv2.polylines(scene, [px_corners], True, (255, 255, 255), 2)
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(scene))
        self.valve.pub.img.publish(img_msg)
    
    def detect_valve_orientation(self, im, bbox=None, HIGHLIGHT=True,
            CannyThreshold1=50, CannyThreshold2=100,
            HoughRho=1, HoughTheta=np.pi/180, HoughThreshold=8,
            HoughMinLineLength=60, HoughMaxLineGap=5):
        """detect_valve_orientation(self, im, bbox=None) -> theta
        Estimate valve orientation using Hough transform:
        Estimate the orientation of the valve in the image within the specified
        bounding box (left, top, right, bottom) in pixels. If bbox==None, the
        entire image is used.
        """
        im_box = im[np.ix_(np.arange(bbox[1], bbox[3]), 
                           np.arange(bbox[0], bbox[2]))]
        im_edges = cv2.Canny(im_box, CannyThreshold1, CannyThreshold2)
        # Zero lines in the borders using a circular mask
        box_height, box_width = im_box.shape
        mask = np.zeros(im_box.shape)
        mask_centre = (int(box_width/2), int(box_height/2))
        mask_radius = int(max((box_height, box_width))/2*1.1)
        cv2.circle(mask, mask_centre, mask_radius, (255, 255, 255), thickness=-1)
        mask[mask>0] = 1
        # mask the 1st and 3rd quadrants since these angles are not possible
        centre_border_px = 0.15*HoughMinLineLength
        cv2.rectangle(mask, (0, box_height),
            (int((box_width/2)-centre_border_px), int((box_height/2)+centre_border_px)),
            (0, 0, 0), thickness=-1)
        cv2.rectangle(mask, (box_width, 0),
            (int((box_width/2)+centre_border_px), int((box_height/2)-centre_border_px)),
            (0, 0, 0), thickness=-1)
        im_edges *= mask
        lines = np.squeeze(cv2.HoughLinesP(im_edges, HoughRho, HoughTheta,
            HoughThreshold, minLineLength=HoughMinLineLength,
            maxLineGap=HoughMaxLineGap))
        # if multiple lines, choose the longest
        if lines.ndim == 1:
            valve_points = lines
        elif lines.ndim == 2:
            pts1 = lines[:, :2]
            pts2 = lines[:, 2:]
            line_lengths = np.sqrt(np.power(pts2 - pts1, 2).sum(axis=1))
            valve_idx = line_lengths.argmax()
            valve_points = lines[valve_idx]
        else:
            return None #raise ValueError
        v_pt1 = valve_points[:2]
        v_pt2 = valve_points[2:]
        delta_xy = v_pt2 - v_pt1
        if HIGHLIGHT and valve_points.shape[0]:
            if not bbox is None:
                v_pt1 += [bbox[0], bbox[1]]
                v_pt2 += [bbox[0], bbox[1]]
            cv2.line(im, tuple(v_pt1), tuple(v_pt2), (0, 0, 0), 6)
        
        # Get the angle using arctan
        valve_orientation = np.arctan2(delta_xy[1], delta_xy[0])
        return valve_orientation
    
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
        remap_camera_name = rospy.resolve_name("stereo_camera")
        rospy.loginfo("Using camera root: " + remap_camera_name)
        remap_image_name = rospy.resolve_name("image_rect")
        rospy.loginfo("Using image topic: " + remap_image_name)
        visual_detector = VisualDetector(rospy.get_name(), 
                                         remap_camera_name, remap_image_name)
        rospy.spin()
    except rospy.ROSInterruptException: pass
