#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
#import std_msgs.msg
import std_srvs.srv
#from auv_msgs.msg import NavSts
from udg_pandora.msg import Detection
import hwu_meta_data.metaclient as metaclient
from detector import GeometricDetector
import cv2
#import code
import objdetect
from cvbridge_wrapper import image_converter
import numpy as np
from featuredetector import cameramodels, image_feature_extractor
import message_filters
from lib.common.misctools import STRUCT, pcl_xyz_cov, approximate_mahalanobis, merge_states
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading
import copy

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


def merge_points(weights, states, covs, merge_threshold=1.):
    num_remaining_components = weights.shape[0]
    merged_wts = []
    merged_sts = []
    merged_cvs = []
    while num_remaining_components:
        max_wt_index = weights.argmax()
        max_wt_state = states[np.newaxis, max_wt_index]
        max_wt_cov = covs[np.newaxis, max_wt_index]
        mahalanobis_fn = approximate_mahalanobis
        mahalanobis_dist = mahalanobis_fn(max_wt_state, max_wt_cov, 
                                          states)
        merge_list_indices = ( np.where(mahalanobis_dist <= 
                                            merge_threshold)[0] )
        retain_idx = ( np.where(mahalanobis_dist > 
                                            merge_threshold)[0] )
        new_wt, new_st, new_cv = merge_states(
                                        weights[merge_list_indices], 
                                        states[merge_list_indices],
                                        covs[merge_list_indices])
        merged_wts += [new_wt]
        merged_sts += [new_st]
        merged_cvs += [new_cv]
        # Remove merged states from the list
        #retain_idx = misctools.gen_retain_idx(self.weights.shape[0], 
        #                                      merge_list_indices)
        weights = weights[retain_idx]
        states = states[retain_idx]
        covs = covs[retain_idx]
        num_remaining_components = weights.shape[0]
    
    return np.array(merged_wts), np.array(merged_sts), np.array(merged_cvs)

class stereo_image_buffer(object):
    def __init__(self, rostopic_camera_root):
        self._image_messages_ = [None, None]
        self._cv_images_ = [None, None]
        self._message_time_ = rospy.Time(0)
        # Convert between sensor_msgs Image anc CV image
        self.ros2cvimg = image_converter()
        #self.subscriptions = STRUCT()
        self._camera_info_topic_ = (
            [rostopic_camera_root+"/left/camera_info",
             rostopic_camera_root+"/right/camera_info"])
        self._image_raw_topic_ = (
            [rostopic_camera_root+"/left/image_raw",
             rostopic_camera_root+"/right/image_raw"])
        
        try:
            self._camera_info_ = (
                [rospy.wait_for_message(cam_info_topic, CameraInfo, 5)
                 for cam_info_topic in self._camera_info_topic_])
        except rospy.ROSException:
            rospy.logfatal("Could not read camera parameters")
            raise rospy.exceptions.ROSException(
                    "Could not read camera parameters")
        
        # List of callbacks and rates
        self._callbacks_ = {}
        self._last_callback_id_ = -1
        self._last_callback_time_ = []
        self._lock_ = threading.Lock()
        
        # Subscribe to left and right images
        self._img_sub_ = [
            message_filters.Subscriber(_sub_image_raw_, Image)
            for _sub_image_raw_ in self._image_raw_topic_]
        self.timesync = message_filters.TimeSynchronizer(self._img_sub_, 5)
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
    
    #get

class VisualDetector:
    def __init__(self, name):
        self.name = name
        self.geometric_detector = GeometricDetector('geometric_detector')
        
        # Create Subscriber
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
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
        
        rostopic_cam_root = "/stereo_front"
        IS_STEREO = True
        # Initialise image buffer
        self.image_buffer = stereo_image_buffer(rostopic_cam_root)
        
        # Initialise panel detector
        template_image_file = "uwsim_panel_template3.png"
        
        panel_corners = np.array([[-0.8, 0.5, 0], [0.8, 0.5, 0], 
                                  [0.8, -0.5, 0], [-0.8, -0.5, 0]])
        panel_update_rate = 4
        self.panel = STRUCT()
        self.panel.br = TransformBroadcaster()
        self.init_panel_detector( 
            template_image_file, panel_corners, IS_STEREO, panel_update_rate)
        self._enablePanelValveDetectionSrv_()
        # Initialise valve detector
        #self.valve = STRUCT()
        #self.valve.detector = objdetect.valve_detector()
        
        # Initialise feature detector for SLAM
        self.slam_features = STRUCT()
        slam_features_update_rate = 1
        num_slam_features = 100
        self.init_slam_feature_detector(slam_features_update_rate, 
                                        num_features=num_slam_features)
        
    def init_panel_detector(self, panel_template_file, panel_corners, 
                            IS_STEREO=False, panel_update_rate=None):
        panel = self.panel
        panel.init = False
        # Read template image
        template_image_file = (
            roslib.packages.find_resource("udg_pandora", panel_template_file))
        if len(template_image_file):
            template_image_file = template_image_file[0]
            template_image = cv2.imread(template_image_file, cv2.CV_8UC1)
        else:
            rospy.logerr("Could not locate panel template")
            raise rospy.exceptions.ROSException(
                    "Could not locate panel template")
        
        # Subscriptions
        if IS_STEREO:
            panel.detector = objdetect.Stereo_detector(
                feat_detector=_default_feature_extractor_)
        else:
            panel.detector = objdetect.Detector(
                feat_detector=_default_feature_extractor_)
        # Set number of features from detector
        panel.detector.set_detector_num_features(2000)
        # Get camera info msg and initialise camera
        cam_info = self.image_buffer.get_camera_info()
        panel.detector.init_camera(*cam_info)
        
        # Set template
        panel.detector.set_template(template_image, panel_corners)
        self.panel.callback_id = None
        self.panel.update_rate = panel_update_rate
        
        # Publish panel position
        panel.pub = metaclient.Publisher('/visual_detector2/valve_panel', Detection, {})
        panel.detection_msg = Detection()
        panel.pose_msg = PoseWithCovarianceStamped()
        panel.pose_msg.header.frame_id = "panel_position"
        panel.pose_msg_pub = metaclient.Publisher('/slam_landmarks/panel_position', PoseWithCovarianceStamped, {})
        # Publish image of detected panel
        self.panel.img_pub = [
            metaclient.Publisher('/visual_detector2/panel_img_l', Image, {}), 
            metaclient.Publisher('/visual_detector2/panel_img_r', Image, {})]
    
    def init_slam_feature_detector(self, update_rate=1, num_features=50, 
                                   hessian_threshold=30000):
        slam_features = self.slam_features
        slam_features.update_rate = update_rate
        
        # Initialise the detector and reset the number of features
        slam_features.camera = cameramodels.StereoCameraFeatureDetector(
            feature_extractor=_default_feature_extractor_, GRID_ADAPTED=True)
        #if _default_feature_extractor_ is image_feature_extractor.Orb:
        #    slam_features.camera._featuredetector_.set_num_features(num_features)
        #elif _default_feature_extractor_ is image_feature_extractor.Surf:
        #    slam_features.camera._featuredetector_.set_hessian_threshold(hessian_threshold)
        #else:
        #    rospy.logfatal("Could not reset the number of features for slam")
        #    raise rospy.ROSException("Could not reset the number of features for slam")
        
        try:
            slam_features.camera._featuredetector_.set_grid_adapted_num_features(num_features)
        except AssertionError as assert_err:
            print assert_err
            rospy.logfatal("Failed to set number of features for slam feature detector")
            raise rospy.ROSException("Could not reset the number of features for slam")
        
        slam_features.camera.fromCameraInfo(*self.image_buffer.get_camera_info())
        
        # Create Publisher
        slam_features.pub = rospy.Publisher("/visual_detector2/features",
                                            PointCloud2)
        slam_features.pcl_helper = pcl_xyz_cov()
        
        # Publish image with detected keypoints
        self.slam_features.img_pub = [
            metaclient.Publisher('/visual_detector2/features_img_l', Image, {}), 
            metaclient.Publisher('/visual_detector2/features_img_r', Image, {})]
        
        # Register the callback
        slam_features.callback_id = (
            self.image_buffer.register_callback(self.detect_slam_features,
                                                slam_features.update_rate))
    
    def enablePanelValveDetectionSrv(self, req):
        #sub = metaclient.Subscriber("VisualDetectorInput", std_msgs.msg.Empty, {}, self.updateImage) 
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
    
    
    def updateImage(self, img):
        pass
    
    def detect_panel(self, *args):
        #self.panel.sub.unregister()
        time_now = rospy.Time.now()
        cvimage = [np.asarray(self.ros2cvimg.cvimagegray(_img_)) for _img_ in args]
        self.panel.detector.detect(*cvimage)
        #self.panel.detector.show()
        panel_detected, panel_position, panel_orientation = (
            self.panel.detector.location())
        panel_rpy = panel_orientation[[2, 0, 1]]
        panel_detected = (panel_detected and 
            self._check_valid_panel_orientation_(panel_rpy))
        panel_orientation_quaternion = quaternion_from_euler(*panel_rpy)
        self.panel.detection_msg.detected = panel_detected
        (self.panel.detection_msg.position.position.x, 
         self.panel.detection_msg.position.position.y,
         self.panel.detection_msg.position.position.z) = panel_position
        (self.panel.detection_msg.position.orientation.x,
         self.panel.detection_msg.position.orientation.y,
         self.panel.detection_msg.position.orientation.z,
         self.panel.detection_msg.position.orientation.w,) = (
             panel_orientation_quaternion)
        
        self.panel.detection_msg.header.stamp = time_now
        self.panel.pub.publish(self.panel.detection_msg)
        # Publish image of detected panel
        scenes = self.panel.detector.get_scene()
        img_msg = [self.ros2cvimg.img_msg(cv2.cv.fromarray(_scene_)) 
                   for _scene_ in scenes]
        for idx in range(len(img_msg)):
            img_msg[idx].header.stamp = time_now
            self.panel.img_pub[idx].publish(img_msg[idx])
        if panel_detected:
            # Panel orientation is screwy - only broadcast the yaw
            # which is panel_orientation[1]
            self.panel.pose_msg.header.stamp = time_now
            self.panel.pose_msg_pub.publish(self.panel.pose_msg)
            self.panel.br.sendTransform(tuple(panel_position),
                panel_orientation_quaternion, 
                time_now, "panel_position", "stereo_front")
        #print "Detected = ", self.panel.detection_msg.detected
        #self._enablePanelValveDetectionSrv_()
    
    def _check_valid_panel_orientation_(self, panel_rpy):
        # if panel_orientation{0,2} not close to zero, return false
        abs_rpy = np.abs(panel_rpy)
        if ((abs_rpy[0] < 0.35 or (np.pi-abs_rpy[0]) < 0.35) and
            (np.pi-abs_rpy[1] < 0.35)):
            return True
        else:
            return False
        #remainder = np.mod(np.pi - np.mod(abs_po, np.pi), np.pi)
        #if np.any(remainder > 0.2):
        #    return False
        #else:
        #    return True
        #return True
        
    def detect_slam_features(self, *args):
        time_now = rospy.Time.now()
        cvimage = [np.asarray(self.ros2cvimg.cvimagegray(_img_)) for _img_ in args]
        points3d, (pts_l, pts_r), (kp_l, kp_r), (desc_l, desc_r) = (
            self.slam_features.camera.points3d_from_img(*cvimage))
        if points3d.shape[0]:
            points_range = np.sqrt((points3d**2).sum(axis=1))
        else:
            points_range = np.empty(0, dtype=np.int32)
        points3d = points3d[points_range <= 3]
        
        # Merge points which are close together
        weights = np.ones(points3d.shape[0])
        covs = np.repeat([np.eye(3)*0.1], points3d.shape[0], axis=0)
        _wts_, points3d_states, points3d_covs = merge_points(weights, points3d, covs)
        print "Detected %s (%s) features" % (points3d_states.shape[0], points3d.shape[0])
        # Convert the points to a pcl message
        if points3d_covs.shape[0]:
            points3d_covs = points3d_covs[:, range(3), range(3)]
        points3d = np.hstack((points3d_states, points3d_covs))
        pcl_msg = self.slam_features.pcl_helper.to_pcl(points3d)
        pcl_msg.header.stamp = time_now
        # set the frame id from the camera
        pcl_msg.header.frame_id = self.slam_features.camera.tfFrame
        # Publish the pcl message
        self.slam_features.pub.publish(pcl_msg)
        pts_l = pts_l.astype(np.int32)
        pts_r = pts_r.astype(np.int32)
        if pts_l.shape[0]:
            cvimage[0][pts_l[:, 1], pts_l[:, 0]] = 255
        if pts_r.shape[0]:
            cvimage[1][pts_r[:, 1], pts_r[:, 0]] = 255
        img_msg = [self.ros2cvimg.img_msg(cv2.cv.fromarray(_scene_)) 
                   for _scene_ in cvimage]
        for idx in range(len(img_msg)):
            img_msg[idx].header.stamp = time_now
            self.slam_features.img_pub[idx].publish(img_msg[idx])
    
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
