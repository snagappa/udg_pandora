# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:28:00 2012

@author: snagappa
"""
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from featuredetector import image_feature_extractor
import phasecorr
import cv2
import numpy as np
import code
#import copy
from scipy import weave
from lib.common.misctools import STRUCT, estimate_rigid_transform_3d
from featuredetector import cameramodels
import tf
#from matplotlib import pyplot
#from lib.common.misctools import cartesian_to_spherical as c2s

# Define services to enable/disable panel detection, valve detection
# (use an internal state?) and chain detection

class Detector(object):
    def __init__(self, feat_detector=image_feature_extractor.orb):
        self._object_ = STRUCT()
        self._object_.template = None
        # Corners of the 3d object
        self._object_.corners_3d = None
        # Corners of the object in the template image
        self._object_.corners_2d = np.empty(0)
        # list of keypoints and descriptors of the template object
        self._object_.keypoints = None
        self._object_.descriptors = None
        # Return status of the FindHomography function
        self._object_.status = False
        # Homography matrix
        self._object_.h_mat = None
        # Number of inliers from homography
        self._object_.num_inliers = 0
        # Status of the inliers
        self._object_.inliers_status = []
        # Specify a threshold to binarise the test scene
        self._object_.intensity_threshold = None
        # Minimum required correspondences to validate localisation
        self._object_.min_correspondences = 10
        # Camera
        self.camera = cameramodels.PinholeCameraFeatureDetector(feat_detector)
        # Rotation matrix and translation vector from solvePnP
        self.obj_rpy = None
        self.obj_trans = None
        self.obj_corners = np.empty(0)
        # Image of the current scene
        self._scene_ = None
        # Set up FLANN matcher
        self.flann_ratio_threshold = 0.75
        
        # Filter for the template and scene
        self.filter_kernel = phasecorr.log_filter_kernel(13)
    
    def init_camera(self, camera_info, *dummy_args):
        self.camera.fromCameraInfo(camera_info)
    
    def set_template(self, template_im=None, corners_3d=None):
        """
        Define a new template and corners of the object in the world.
        """
        # Detect features using the FINE preset
        if not template_im is None:
            template_im = self.process_images((template_im,))[0]
            self._object_.template = template_im
            (self._object_.keypoints, self._object_.descriptors) = \
                self.camera.get_features(self._object_.template)
            height, width = self._object_.template.shape[:2]
            self._object_.corners_2d = np.float32([[0, 0], 
                                                   [width, 0], 
                                                   [width, height], 
                                                   [0, height]])
        if not corners_3d is None:
            assert_err_msg = "corners must a Nx3 ndarray, N>=4"
            assert type(corners_3d) is np.ndarray, assert_err_msg
            assert (corners_3d.ndim==2) and \
                   (corners_3d.shape[1]==3) and \
                   (corners_3d.shape[0]>3), assert_err_msg
            self._object_.corners_3d = corners_3d
    
    def process_images(self, images=()):
        assert type(images) is tuple, "Images must be a tuple"
        #return [self._binarise_(cv2.filter2D(_im_, -1, self.log_kernel), 
        #                        self._object_.intensity_threshold) 
        #        for _im_ in images]
        return list(images)
    
    def _binarise_(self, im, intensity_threshold):
        if not intensity_threshold is None:
            im[im > intensity_threshold] = 255
            im[im <= intensity_threshold] = 0
        return im
    
    def _sharpen_(self, im, filter_size=(5, 5), alpha=1.5, beta=-0.5):
        sm_im = cv2.GaussianBlur(im, filter_size, 0)
        return cv2.addWeighted(im, alpha, sm_im, beta, 1.0)
        
    def _detect_(self, im_scene):
        if self._object_.template is None:
            print "object template is not set!"
            return None
        #self._scene_ = copy.copy(im_scene)
        # Filter the scene
        proc_im_scene = self.process_images((im_scene,))[0]
        (keypoints_scene, descriptors_scene) = (
            self.camera.get_features(proc_im_scene))
        dam_result = self.camera.detect_and_match(self._object_.keypoints,
                                                  self._object_.descriptors,
                                                  keypoints_scene,
                                                  descriptors_scene,
                                                  self.flann_ratio_threshold)
        pts_1, pts_2 = dam_result[0:2]
        status, h_mat, num_inliers, inliers_status = (
            self.camera.find_homography(pts_1, pts_2, 
            min_inliers=self._object_.min_correspondences))
        return status, h_mat, num_inliers, inliers_status
    
    def detect(self, im_scene, *dummy_args):
        """
        detect(self, im_scene) -> None
        Detects object by matching features and calculating the homography 
        matrix from the template and current scene.
        """
        self._scene_ = im_scene
        status, h_mat, num_inliers, inliers_status = self._detect_(im_scene)
        print '%d / %d  inliers/matched' % (num_inliers, len(inliers_status))
        self._object_.status = status
        self._object_.h_mat = h_mat
        self._object_.num_inliers = num_inliers
        self._object_.inliers_status = inliers_status
        return status
        
    def homography(self):
        """
        Return the estimated homography matrix
        """
        return self._object_.h_mat
    
    def location(self):
        """
        Return detected (bool), relative position (x, y, z) and 
        orientation (RPY) of the object.
        """
        # From http://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # "T is not the position of the camera. It is the position of 
        # the origin of the world coordinate system expressed in 
        # coordinates of the camera-centered coordinate system. The 
        # position, C, of the camera expressed in world coordinates is 
        # C = -R^{-1}T = -R^T T (since R is a rotation matrix)."
        #camera_centre = np.dot(-(cv2.Rodrigues(rvec)[0].T), tvec)
        detected = False
        if self._object_.status:
            # Project the corners of the template onto the scene
            self.obj_corners = cv2.perspectiveTransform(
                self._object_.corners_2d.reshape(1, -1, 2), 
                self._object_.h_mat).reshape(-1, 2)
            # Solve perspective n-point
            retval, rvec, tvec = cv2.solvePnP(self._object_.corners_3d, 
                self.obj_corners, self.camera.camera_matrix(), np.empty(0))
            # Convert the rotation vector to RPY
            r_mat = cv2.Rodrigues(rvec)[0]
            self.obj_rpy = np.asarray(tf.transformations.euler_from_matrix(r_mat))
            self.obj_trans = tvec
            obj_range = np.linalg.norm(self.obj_trans)
            if not retval or (not 0.25 < obj_range < 5):
                print "Too close/far for reliable estimation"
            else:
                detected = True
                # Plot outline on image
                corners = self.obj_corners.astype(np.int32)
                cv2.polylines(self.get_scene(0), [corners], 
                              True, (255, 255, 255), 4)
        else:
            self.obj_trans = np.zeros(3)
            self.obj_rpy = np.zeros(3)
        return detected, self.obj_trans, self.obj_rpy
    
    def get_scene(self, idx=None):
        if not idx is None:
            return (self._scene_,)[idx]
        else:
            return (self._scene_,)
    
    def set_detector_num_features(self, num_features):
        self.camera.set_detector_num_features(num_features)
    
    #def show(self):
    #    cv2.namedWindow("panel-detect")
    #    if not self._object_.h_mat is None:
    #        print "Detected the panel!"
    #    #else:
    #    #    print "No detection."
    #    if not self._scene_ is None:
    #        cv2.imshow("panel-detect", self._scene_)
            

class Stereo_detector(Detector):
    def __init__(self, feat_detector=image_feature_extractor.orb):
        # Initialise the detector using the parent class constructor
        super(Stereo_detector, self).__init__(feat_detector=feat_detector)
        self.camera = cameramodels.StereoCameraFeatureDetector(feat_detector)
        # Additional homography for second camera
        self.template_z_offset = 0
        self._object_.keypoints_2d = None
        self._object_.keypoints_3d = None
        self._object_.affine_3d = None
    
    def init_camera(self, camera_info_left, camera_info_right):
        self.camera.fromCameraInfo(camera_info_left, camera_info_right)
        # Additional parameters for the stereo detection
        proj_matrix = self.camera.projection_matrix()
        self.camera_half_baseline = (
            np.abs(proj_matrix[1, 0, 3]/proj_matrix[1, 0, 0])/2)
    
    def set_template(self, template_im=None, corners_3d=None):
        super(Stereo_detector, self).set_template(template_im, corners_3d)
        if len(self._object_.keypoints) == 0:
            rospy.logfatal("No keypoints detected in template image")
            raise rospy.ROSException("No keypoints detected in template image")
        self._object_.keypoints_2d = np.array(
            [_kp_.pt for _kp_ in self._object_.keypoints])
        keypoints_normalised = (self._object_.keypoints_2d/
                                [template_im.shape[1], template_im.shape[0]])
        keypoints_normalised -= 0.5
        # Find size of the template by taking the difference of the corners
        template_size = np.max(np.abs(np.diff(corners_3d, axis=0)), axis=0)
        keypoints_3d = np.hstack((keypoints_normalised, 
            np.zeros((keypoints_normalised.shape[0], 1))))
        keypoints_3d *= template_size
        self._object_.keypoints_3d = keypoints_3d
    
    def detect(self, im_scene_l, im_scene_r):
        """
        detect(self, im_scene) -> None
        Detects object by matching features and calculating the homography 
        matrix from the template and current scene.
        """
        # Set affine transformation and status to false
        self._object_.status = [False, False]
        self._object_.h_mat = [None, None]
        self._object_.num_inliers = [0, 0]
        self._object_.inliers_status = [None, None]
        self.obj_rpy = np.zeros(3)
        self.obj_trans = np.zeros(3)
        self.obj_corners = np.empty(0)
        
        # Threshold the image
        self._scene_ = self.process_images((im_scene_l, im_scene_r))
        
        if self._object_.template is None:
            print "object template is not set!"
            return None
        
        status_l, h_mat_l, num_inliers_l, inliers_status_l = self._detect_(im_scene_l)
        status_r, h_mat_r, num_inliers_r, inliers_status_r = self._detect_(im_scene_r)
        self._object_.status = [status_l, status_r]
        self._object_.h_mat = [h_mat_l, h_mat_r]
        self._object_.num_inliers = [num_inliers_l, num_inliers_r]
        self._object_.inliers_status = [inliers_status_l, inliers_status_r]
    
    def location(self):
        """
        Return detected (bool), relative position (x, y, z) and 
        orientation (RPY) of the object.
        """
        status = self._object_.status
        h_mat = self._object_.h_mat
        num_inliers = np.asarray(self._object_.num_inliers)
        #inliers_status = self._object_.inliers_status
        obj_corners = []
        obj_trans = []
        obj_rpy = []
        for idx in range(2):
            if status[idx]:
                _obj_corners_ = cv2.perspectiveTransform(
                    self._object_.corners_2d.reshape(1, -1, 2), 
                    h_mat[idx]).reshape(-1, 2)
                # Solve perspective n-point
                _retval_, _rvec_, _tvec_ = cv2.solvePnP(
                    self._object_.corners_3d, _obj_corners_, 
                    self.camera.camera_matrix(), np.empty(0))
                # Convert the rotation vector to RPY
                _r_mat_ = cv2.Rodrigues(_rvec_)[0]
                _rpy_ = np.asarray(tf.transformations.euler_from_matrix(_r_mat_))
                # Check if object in valid range
                obj_range = np.linalg.norm(_tvec_)
                if not _retval_ or (not 0.25 < obj_range < 5):
                    status[idx] = False
                else:
                    # Plot the bounding box
                    corners = _obj_corners_.astype(np.int32)
                    cv2.polylines(self.get_scene(idx), [corners], 
                                  True, (255, 255, 255), 4)
            else:
                _obj_corners_ = np.zeros((1, 3))
                _rpy_ = np.zeros(3)
                _tvec_ = np.zeros(3)
            obj_corners.append(_obj_corners_)
            obj_trans.append(np.squeeze(_tvec_))
            obj_rpy.append(_rpy_)
        
        # Convert the translation from the right image to the left
        tf_obj_trans = self.camera.left.from_world_coords(
            self.camera.right.to_world_coords(obj_trans[1][np.newaxis]))[0]
        detected = status[0] or status[1]
        # Compute the weighted average of the pose
        total_inliers = (num_inliers*status).sum()
        try:
            position = np.squeeze((obj_trans[0]*status[0]*num_inliers[0] +
                    tf_obj_trans*status[1]*num_inliers[1])/total_inliers)
        except:
            print "tf failed!"
            return 0, np.zeros(3), np.zeros(3)
        orientation = (obj_rpy[0]*status[0]*num_inliers[0] +
                       obj_rpy[1]*status[1]*num_inliers[1])/total_inliers
        return detected, position, orientation
    
    def get_scene(self, idx=None):
        if not idx is None:
            return self._scene_[idx]
        else:
            return self._scene_
    

def _estimateAffine3D_(src_points, dst_points):
    """
    (retval, affine_3d, inliers) = _estimateAffine3D_(np_pts_from, np_pts_to)
    """
    err_msg = "points must be a Nx3 ndarray"
    assert np.ndim(src_points) == np.ndim(dst_points) == 2, err_msg
    assert src_points.shape[1] == dst_points.shape[1] == 3, err_msg
    assert ((src_points.shape[0] > 0) and
            (dst_points.shape[0] > 0)), err_msg
    if not src_points.dtype == np.float32:
        print "src_points should have dtype=np.float32, converting..."
        np_points_from = src_points.astype(np.float32)
    else:
        np_points_from = src_points
    if not dst_points.dtype == np.float32:
        print "dst_points should have dtype=np.float32, converting..."
        np_points_to = dst_points.astype(np.float32)
    else:
        np_points_to = dst_points
    
    affine_3d = np.zeros((3, 4), dtype=np.float64)
    inliers = []
    num_inliers = 0
    retval = 0
    
    header_code = """
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv/cvaux.h>
#include <string>
#include <iostream>
"""
    fn_code = """
int num_points_from, num_points_to, result;

// Number of points is the number of rows
num_points_from = Nnp_points_from[0];
num_points_to = Nnp_points_to[0];

// Create affine matrix
cv::Mat aff(3, 4, CV_64F, affine_3d);

// vector of inliers
std::vector<uchar> inliers;

// Prepare two sets of 3D points
cv::Mat mat_points_from(num_points_from, 3, CV_32F, np_points_from);
cv::Mat mat_points_to(num_points_to, 3, CV_32F, np_points_to);
// Convert the Mat to vectors of Point3f, otherwise RANSAC will fail
std::vector<cv::Point3f> points_from = cv::Mat_<cv::Point3f>(mat_points_from);
std::vector<cv::Point3f> points_to = cv::Mat_<cv::Point3f>(mat_points_to);

result = cv::estimateAffine3D(points_from, points_to, aff, inliers);
num_inliers = inliers.size();
retval = result;
"""
    ros_root = "/opt/ros/fuerte/"
    python_vars = ["np_points_from", "np_points_to", "affine_3d", 
                   "retval", "num_inliers"]
    weave.inline(fn_code, python_vars, support_code=header_code,
                 libraries=["opencv_calib3d"],
                 library_dirs=[ros_root+"lib"],
                 include_dirs=[ros_root+"include", ros_root+"include/opencv"],
                 extra_compile_args=["-O2 -g"], verbose=1)
    inliers = [num_inliers]
    return not(retval), affine_3d, inliers