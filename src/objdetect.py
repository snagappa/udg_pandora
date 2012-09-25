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
from matplotlib import pyplot

# Define services to enable/disable panel detection, valve detection
# (use an internal state?) and chain detection

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

class STRUCT(object): pass

class FlannMatcher_wrapper(object):
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS.copy()
        self.NEW_FLANN_MATCHER = False
         # Use the cv2 FlannBasedMatcher if available
        try:
            self._flann_ = cv2.FlannBasedMatcher(self.PARAMS, {})  # bug : need to pass empty dict (#1329)        
            self.NEW_FLANN_MATCHER = True
        except AttributeError, ae:
            rospy.loginfo(ae)
            rospy.loginfo("Could not initialise FlannBasedMatcher, using fallback")
    
    def knnMatch(self, queryDescriptors, trainDescriptors, k, mask=None, compactResult=None):
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
                        this_match_dist, this_match_idx = zip(*this_match_dist_idx)
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
            idx2, distance = self._flann_.knnSearch(queryDescriptors, k, params = {}) # bug: need to provide empty dict
        idx1 = np.arange(len(queryDescriptors))
        return idx1, idx2, distance
        

class detector(object):
    def __init__(self, template=None, 
                 feat_detector=image_feature_extractor.orb, 
                 detector_preset=image_feature_extractor.DETECTOR_PRESET.MED):
        self._object_ = STRUCT()
        self._object_.template = None
        self._object_.corners = np.empty(0)
        self._object_.keypoints = None
        self._object_.descriptors = None
        self._object_.H = None
        self._object_.status = None
        self._scene_ = None
        self._detector_ = feat_detector(detector_preset)
        # Set up FLANN matcher
        self._flann_ = STRUCT()
        self._flann_.r_threshold = 0.6
        
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
        self._flann_.matcher = FlannMatcher_wrapper(self._flann_.PARAMS)
        if not template is None:
            self.set_template(template.copy())
        
    
    def set_template(self, template_im):
        # Detect features using the FINE preset
        self._object_.template = template_im
        template_detector = self._detector_.__class__(
            image_feature_extractor.DETECTOR_PRESET.FINE)
        (self._object_.keypoints, self._object_.descriptors) = \
            template_detector.get_features(self._object_.template)
        h1, w1 = self._object_.template.shape[:2]
        self._object_.corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            
    def _detect_and_match_(self, obj_kp, obj_desc, scene_kp, scene_desc, ratio=0.75):
        idx1, idx2, distance = self._flann_.matcher.knnMatch(obj_desc, scene_desc, 2)
        # Use only good matches
        mask = distance[:,0] < (distance[:,1] * ratio)
        mask[idx2[:,1]==-1] = False
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
        if self._object_.template is None:
            print "object template is not set!"
            return None
        self._scene_ = copy.copy(im_scene)
        
        (keypoints_scene, descriptors_scene) = (
            self._detector_.get_features(self._scene_))
        if not keypoints_scene:
            H = None
            status = None
        else:
            p1, p2, kp_pairs = self._detect_and_match_(self._object_.keypoints,
                                                       self._object_.descriptors,
                                                       keypoints_scene,
                                                       descriptors_scene,
                                                       self._flann_.r_threshold)
            
            if len(p1) >= 15:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                
                if H is not None:
                    corners = np.int32( cv2.perspectiveTransform(
                        self._object_.corners.reshape(1, -1, 2), H).reshape(-1, 2))
                        #+ (self._object_.template.shape[1], 0) )
                    cv2.polylines(self._scene_, [corners], True, (255, 255, 255), 4)
            
            else:
                H, status = None, None
                #print '%d matches found, not enough for homography estimation' % len(p1)
        self._object_.H = H
        self._object_.status = status
        #return H, status
        
    def homography(self):
        return self._object_.H
        
    def show(self):
        cv2.namedWindow("panel-detect")
        if not self._object_.H is None:
            print "Detected the panel!"
        #else:
        #    print "No detection."
        if not self._scene_ is None:
            cv2.imshow("panel-detect", self._scene_)
            
