# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:28:00 2012

@author: snagappa
"""
import roslib
roslib.load_manifest('udg_pandora')
import rospy
import image_feature_extractor
import cv2
import numpy as np
import code
import copy
#from matplotlib import pyplot
#from lib.common.misctools import cartesian_to_spherical as c2s

# Define services to enable/disable panel detection, valve detection
# (use an internal state?) and chain detection

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def uwsim_cam_calib(res_x=640, res_y=480):
    # See Section 4.2 of "Programming Computer Vision with Python" by 
    # Jan Erik Solem
    focal_x = 572*res_x/640
    focal_y = 356*res_y/480
    camera_matrix = np.diag([focal_x, focal_y, 1])
    camera_matrix[0, 2] = 0.5*res_x
    camera_matrix[1, 2] = 0.5*res_y
    return camera_matrix.astype(np.float)

class STRUCT(object): pass

class FlannMatcher(object):
    """
    Wrapper class for using the Flann matcher. Attempts to use the new 
    FlannBasedMatcher interface, but uses the fallback flann_Index if this is
    unavailable.
    """
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS.copy()
        self.NEW_FLANN_MATCHER = False
        try:
            # Use the cv2 FlannBasedMatcher if available
            # bug : need to pass empty dict (#1329)        
            self._flann_ = cv2.FlannBasedMatcher(self.PARAMS, {})  
            self.NEW_FLANN_MATCHER = True
        except AttributeError, ae:
            rospy.loginfo(ae)
            rospy.loginfo(
                "Could not initialise FlannBasedMatcher, using fallback")
    
    def knnMatch(self, queryDescriptors, trainDescriptors, k, mask=None, 
                 compactResult=None):
        """
        knnMatch(queryDescriptors, trainDescriptors, k, mask=None, 
                 compactResult=None) -> idx1, idx2, distance
        Returns k best matches between queryDescriptors indexed by idx1 and 
        trainDescriptors indexed by idx2. Distance between the descriptors is
        given by distance, a Nxk ndarray.
        """
        if self.NEW_FLANN_MATCHER:
            matches = self._flann_.knnMatch(queryDescriptors, 
                                            trainDescriptors, 
                                            k = 2) #2
            # Extract the distance and indices from the list of matches
            num_descriptors = len(queryDescriptors)
            # Default distance is one
            distance = np.ones((num_descriptors, k))
            idx2 = np.zeros((num_descriptors, k), dtype=np.int)
            try:
                for m_count in range(num_descriptors):
                    this_match_dist_idx = [(_match_.distance, _match_.trainIdx)
                        for _match_ in matches[m_count]]
                    # Only proceed if we have a match, otherwise leave defaults
                    if this_match_dist_idx:
                        (this_match_dist, 
                         this_match_idx) = zip(*this_match_dist_idx)
                        this_match_len = len(this_match_dist)
                        distance[m_count, 0:this_match_len] = this_match_dist
                        idx2[m_count, 0:this_match_len] = this_match_idx
                        if this_match_len < k:
                            distance[m_count, this_match_len:] = (
                                distance[m_count, this_match_len-1])
                            idx2[m_count, this_match_len:] = (
                                idx2[m_count, this_match_len-1])
            except:
                print "error occurred"
                code.interact(local=locals())
        else:
            self._flann_ = cv2.flann_Index(trainDescriptors, self.PARAMS)
            # Perform nearest neighbours search
            # bug: need to provide empty dict for params
            idx2, distance = self._flann_.knnSearch(queryDescriptors, k, 
                                                    params={})
        idx1 = np.arange(len(queryDescriptors))
        return idx1, idx2, distance
        

class Detector(object):
    def __init__(self, template=None, corners_3d=None, 
                 camera_matrix=uwsim_cam_calib(),
                 feat_detector=image_feature_extractor.orb, 
                 detector_preset=image_feature_extractor.DETECTOR_PRESET.MED):
        self._object_ = STRUCT()
        self._object_.template = None
        # Corners of the 3d object
        self._object_.corners_3d = None
        # Corners of the object in the template image
        self._object_.corners_2d = np.empty(0)
        # list of keypoints and descriptors of the template object
        self._object_.keypoints = None
        self._object_.descriptors = None
        # Homography matrix
        self._object_.h_mat = None
        # Return status of the FindHomography function
        self._object_.status = None
        # Specify a threshold to binarise the test scene
        self._object_.scene_intensity_threshold = None
        # Camera matrix
        self.camera_matrix = camera_matrix.astype(np.float)
        # Image of the current scene
        self._scene_ = None
        # Create the feature detector
        self._detector_ = feat_detector(detector_preset)
        # Set up FLANN matcher
        self._flann_ = STRUCT()
        self._flann_.r_threshold = 0.6
        
        # Set up the parameters for FLANN matching
        # cv2 norms may not be defined. Use workaround if necessary
        try:
            detector_norm = self._detector_.NORM
            norm_hamming = cv2.NORM_HAMMING
        except:
            detector_norm = self._detector_.DESCRIPTOR_IS_BINARY
            norm_hamming = True
        if detector_norm == norm_hamming: # Use LSH if binary descriptor
            self._flann_.PARAMS = dict(algorithm = FLANN_INDEX_LSH,
                                       table_number = 6, # 12
                                       key_size = 12,     # 20
                                       multi_probe_level = 1) #2
        else:
            self._flann_.PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, 
                                       trees = 5)
        # Initialise the flann matcher
        self._flann_.matcher = FlannMatcher(self._flann_.PARAMS)
        # Initialise the template
        self.set_template(template, corners_3d)
        
    
    def set_template(self, template_im=None, corners_3d=None):
        """
        Define a new template and corners of the object in the world.
        """
        # Detect features using the FINE preset
        if not template_im is None:
            self._object_.template = template_im
            template_detector = self._detector_.__class__(
                image_feature_extractor.DETECTOR_PRESET.FINE)
            (self._object_.keypoints, self._object_.descriptors) = \
                template_detector.get_features(self._object_.template)
            height, width = self._object_.template.shape[:2]
            self._object_.corners_2d = np.float32([[0, 0], 
                                                   [width, 0], 
                                                   [width, height], 
                                                   [0, height]])
        if not corners_3d is None:
            assert_err_msg = "corners must a Nx3 ndarray, N>3"
            assert type(corners_3d) is np.ndarray, assert_err_msg
            assert (corners_3d.ndim==2) and \
                   (corners_3d.shape[1]==3) and \
                   (corners_3d.shape[0]>3), assert_err_msg
            self._object_.corners_3d = corners_3d
            
    def _detect_and_match_(self, obj_kp, obj_desc, scene_kp, scene_desc, 
                           ratio=0.75):
        idx1, idx2, distance = self._flann_.matcher.knnMatch(obj_desc, 
                                                             scene_desc, 2)
        # Use only good matches
        mask = distance[:, 0] < (distance[:, 1] * ratio)
        mask[idx2[:,1] == -1] = False
        valid_idx1 = idx1[mask]
        valid_idx2 = idx2[mask, 0]
        match_kp1, match_kp2 = [], []
        for (_idx1_, _idx2_) in zip(valid_idx1, valid_idx2):
            match_kp1.append(obj_kp[_idx1_])
            match_kp2.append(scene_kp[_idx2_])
        p1 = np.float32([kp.pt for kp in match_kp1])
        p2 = np.float32([kp.pt for kp in match_kp2])
        kp_pairs = zip(match_kp1, match_kp2)
        return p1, p2, kp_pairs
    
    def detect(self, im_scene):
        """
        Calculates the homography matrix using the template and current scene.
        """
        if self._object_.template is None:
            print "object template is not set!"
            return None
        self._scene_ = copy.copy(im_scene)
        # Threshold the image
        #intensity_threshold = self._object_.scene_intensity_threshold
        #self._scene_[self._scene_ > intensity_threshold] = 255
        #self._scene_[self._scene_ <= intensity_threshold] = 0
        (keypoints_scene, descriptors_scene) = (
            self._detector_.get_features(self._scene_))
        if not keypoints_scene:
            h_mat = None
            status = None
        else:
            (p1, p2, 
             kp_pairs) = self._detect_and_match_(self._object_.keypoints,
                                                 self._object_.descriptors,
                                                 keypoints_scene,
                                                 descriptors_scene,
                                                 self._flann_.r_threshold)
            # Compute homography only if we have at least 10 matches
            if len(p1) >= 10:
                h_mat, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                inliers = np.sum(status)
                print '%d / %d  inliers/matched' % (inliers, len(status))
                # Homography is valid only if we have at least 10 inliers
                if (h_mat is None) or (inliers < 10):
                    h_mat, status = None, None
            else:
                h_mat, status = None, None
                #print '%d matches found,\
                #   not enough for homography estimation' % len(p1)
        self._object_.h_mat = h_mat
        self._object_.status = status
        #return h_mat, status
        
    def homography(self):
        """
        Return the estimated homography matrix
        """
        return self._object_.h_mat
        
    def location(self):
        """
        Return detected (bool) and relative position (x, y, z) of the object.
        """
        detected = False
        position = np.zeros(3)
        # Evaluate the location only if the homography is valid
        if not self._object_.h_mat is None:
            # Compute the location if the object corners are known in world
            # co-ordinates
            if not self._object_.corners_3d is None:
                # Project the corners of the template onto the scene
                corners_float = cv2.perspectiveTransform(
                    self._object_.corners_2d.reshape(1, -1, 2), 
                    self._object_.h_mat).reshape(-1, 2)
                #corners = corners_float.astype(np.int32)
                #cv2.polylines(self._scene_, [corners], 
                #              True, (255, 255, 255), 4)
            
                retval, rvec, tvec = cv2.solvePnP(self._object_.corners_3d, 
                                                  corners_float, 
                                                  self.camera_matrix, 
                                                  np.empty(0))
                #sph_tvec = (c2s(tvec).T*[1, 180/np.pi, 180/np.pi])[0]
                obj_range = np.linalg.norm(tvec)
                if not retval or (not 0.25 < obj_range < 5):
                    print "Too close/far for reliable estimation"
                else:
                    detected = True
                    position = tvec
                print "Relative panel position (x,y,z) = ", str(tvec.T)
                #print "Relative panel position (r,az,el) = ", str(sph_tvec)
        return detected, position
            
    #def show(self):
    #    cv2.namedWindow("panel-detect")
    #    if not self._object_.h_mat is None:
    #        print "Detected the panel!"
    #    #else:
    #    #    print "No detection."
    #    if not self._scene_ is None:
    #        cv2.imshow("panel-detect", self._scene_)
            
