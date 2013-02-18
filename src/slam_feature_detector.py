#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import hwu_meta_data.metaclient as metaclient
import cv2
import code
import numpy as np
import message_filters
from lib.misctools import STRUCT, pcl_xyz_cov, approximate_mahalanobis, \
    merge_states, image_converter, pcl_xyz
from lib import cameramodels, image_feature_extractor
import threading
import copy
import pickle

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

FOV_FAR = 3.0

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
        
        # ROS image message to cvimage convertor
        self.ros2cvimg = image_converter()
        
        # Initialise image buffer
        self.image_buffer = stereo_image_buffer(self.rostopic_cam_root,
                                                self.image_sub_topic)
        
        # Initialise feature detector for SLAM
        self.slam_features = STRUCT()
        slam_features_update_rate = None
        num_slam_features = 40
        self.init_slam_feature_detector(slam_features_update_rate,
                                        num_features=num_slam_features,
                                        hessian_threshold=50)
        print "Completed initialisation"
    
    def init_slam_feature_detector(self, update_rate=1, num_features=50,
                                   hessian_threshold=None):
        slam_features = self.slam_features
        slam_features.update_rate = update_rate
        
        # Initialise the detector and reset the number of features
        slam_features.camera = cameramodels.StereoCameraFeatureDetector(
            feature_extractor=image_feature_extractor.Surf, GRID_ADAPTED=False)
        #if _default_feature_extractor_ is image_feature_extractor.Orb:
        #    slam_features.camera._featuredetector_.set_num_features(num_features)
        #elif _default_feature_extractor_ is image_feature_extractor.Surf:
        #    slam_features.camera._featuredetector_.set_hessian_threshold(hessian_threshold)
        #else:
        #    rospy.logfatal("Could not reset the number of features for slam")
        #    raise rospy.ROSException("Could not reset the number of features for slam")
        
        # Set the hessian threshold if possible
        set_hessian_threshold = getattr(slam_features.camera._featuredetector_,
                                        "set_hessian_threshold", None)
        if not set_hessian_threshold is None and not hessian_threshold is None:
            print "Setting hessian threhosld to %s" % hessian_threshold
            set_hessian_threshold(hessian_threshold)
        slam_features.camera._featuredetector_.set_nOctaves(7)
        slam_features.camera._featuredetector_.make_grid_adapted()
        
        try:
            slam_features.camera._featuredetector_.set_num_features(num_features)
        except (AssertionError, UnboundLocalError) as assert_err:
            print assert_err
            rospy.logerr("Failed to set number of features for slam feature detector")
            raise rospy.ROSException("Could not reset the number of features for slam")
        
        slam_features.camera.fromCameraInfo(*self.image_buffer.get_camera_info())
        slam_features.camera.set_near_far_fov(fov_far=FOV_FAR)
        # Create Publisher
        slam_features.pub = rospy.Publisher("/visual_detector2/features",
                                            PointCloud2)
        slam_features.pub2 = rospy.Publisher("/visual_detector2/features_xyz",
                                            PointCloud2)
        slam_features.pcl_helper = pcl_xyz_cov()
        slam_features.pcl_helper2 = pcl_xyz()
        
        # Publish image with detected keypoints
        self.slam_features.img_pub = [
            metaclient.Publisher('/visual_detector2/features_img_l', Image, {}),
            metaclient.Publisher('/visual_detector2/features_img_r', Image, {})]
        
        # Register the callback
        slam_features.callback_id = (
            self.image_buffer.register_callback(self.detect_slam_features,
                                                slam_features.update_rate))
    
    def detect_slam_features(self, *args):
        #time_now = args[0].header.stamp
        time_now = rospy.Time.now()
        cvimage = [np.asarray(self.ros2cvimg.cvimagegray(_img_)).copy() for _img_ in args]
        points3d, (pts_l, pts_r), (kp_l, kp_r), (desc_l, desc_r) = (
            self.slam_features.camera.points3d_from_img(*cvimage,
                ratio_threshold=0.6, image_margins=(64, 64, 64, 64)))
        if points3d.shape[0]:
            points_range = np.sqrt((points3d**2).sum(axis=1))
        else:
            #print "no features found"
            points_range = np.empty(0, dtype=np.int32)
        #print "Found %s features at stage 1" % points3d.shape[0]
        #code.interact(local=locals())
        #print "Average distance = %s" % np.mean(points3d, axis=0)
        points3d = points3d[points_range <= FOV_FAR]
        points_range = points_range[points_range <= FOV_FAR]
        #print "Ignoring features beyond %s m; remaining = %s" % (FOV_FAR, points3d.shape[0])
        # Merge points which are close together
        weights = np.ones(points3d.shape[0])
        x_cov_base = (8e-2)**2
        y_cov_base = (8e-2)**2
        z_cov_base = (8e-2)**2
        points3d_scale = np.hstack((points_range[:, np.newaxis], 
                                    points_range[:, np.newaxis], 
                                    points_range[:, np.newaxis]))
        covs = np.array([x_cov_base, y_cov_base, z_cov_base])*points3d_scale[:, np.newaxis, :]*np.eye(3)[np.newaxis]
        _wts_, points3d_states, points3d_covs = merge_points(weights, points3d, covs, merge_threshold=0.1)
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
        
        pcl_msg2 = self.slam_features.pcl_helper2.to_pcl(points3d_states)
        pcl_msg2.header.stamp = time_now
        pcl_msg2.header.frame_id = "/world"
        self.slam_features.pub2.publish(pcl_msg2)
        
        pts_l = pts_l.astype(np.int32)
        pts_r = pts_r.astype(np.int32)
        (pts_l_reproj, pts_r_reproj) = self.slam_features.camera.project3dToPixel(points3d_states)
        pts_l_reproj = pts_l_reproj.astype(np.int)
        pts_r_reproj = pts_r_reproj.astype(np.int)
        if pts_l.shape[0]:
            cvimage[0][pts_l[:, 1], pts_l[:, 0]] = 255
            for landmark in pts_l_reproj:
                cv2.circle(cvimage[0], tuple(landmark), 5, (0, 0, 0), -1)
        if pts_r.shape[0]:
            cvimage[1][pts_r[:, 1], pts_r[:, 0]] = 255
            for landmark in pts_r_reproj:
                cv2.circle(cvimage[1], tuple(landmark), 5, (0, 0, 0), -1)
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
        rospy.init_node('slam_feature_detector')
        remap_camera_name = rospy.resolve_name("stereo_camera")
        rospy.loginfo("Using camera root: " + remap_camera_name)
        remap_image_name = rospy.resolve_name("image_rect")
        rospy.loginfo("Using image topic: " + remap_image_name)
        visual_detector = VisualDetector(rospy.get_name(), 
                                         remap_camera_name, remap_image_name)
        rospy.spin()
    except rospy.ROSInterruptException: pass
