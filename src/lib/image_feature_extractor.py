# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:33:17 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('udg_pandora')
import cv2
from misctools import STRUCT, FlannMatcher
#import collections
import numpy as np
import copy
import code


class DETECTOR_PRESET:
    FINE, MED, COARSE = range(3)


DETECTORS = ("SIFT", "SURF", "FAST", "MSER", "ORB")
BINARY_DESCRIPTORS = ("ORB", "FREAK")
DESCRIPTORS = ("SIFT", "SURF") + BINARY_DESCRIPTORS


class _feature_detector_(object):
    def __init__(self, feature_detector, feature_descriptor, 
                 GRID_ADAPTED=False):
        """
        _feature_detector_(detector<str>, descriptor<str>) 
            -> _feature_detector_ object
        """
        assert ((type(feature_detector) == str) and 
                (type(feature_descriptor) == str)), "Detector and descriptor must be strings"
        feature_detector = feature_detector.upper()
        feature_descriptor = feature_descriptor.upper()
        assert feature_detector in DETECTORS, "Detector must be one of"+str(DETECTORS)
        assert feature_descriptor in DESCRIPTORS, "Detector must be one of"+str(DESCRIPTORS)
        # Storage for keypoints and descriptors
        self._keypoints_ = None
        self._descriptors_ = None
        # Type of norm used for the descriptor
        self.NORM = None
        self.DESCRIPTOR_IS_BINARY = -1
        
        # Type of detector and descriptor used
        self.type = STRUCT()
        self.type.feature_detector = feature_detector
        self.type.feature_descriptor = feature_descriptor
        self._detector_init_()
        self.GRID_ADAPTED = False
        if GRID_ADAPTED:
            self.make_grid_adapted()
    
    def _detector_init_(self):
        # Initialise the feature detector
        self._detector_ = cv2.FeatureDetector_create(self.type.feature_detector)
        
        # Initialise the descriptor extractor
        self._descriptor_extractor_ = (
            cv2.DescriptorExtractor_create(self.type.feature_descriptor))
        # lambda function to compute the keypoints and descriptors
        self._detect_ = lambda im, mask: \
            self._descriptor_extractor_.compute(im, 
            self._detector_.detect(im, mask))
        # Set the norm so that the appropriate FLANN matching rules can be set
        if self.type.feature_descriptor in BINARY_DESCRIPTORS:
            try:
                self.NORM = cv2.NORM_HAMMING
            except AttributeError:
                self.NORM = None
            self.DESCRIPTOR_IS_BINARY = True
        else:
            try:
                self.NORM = cv2.NORM_L2
            except AttributeError:
                self.NORM = None
            self.DESCRIPTOR_IS_BINARY = False
        
    def make_grid_adapted(self, FORCE=False):
        if self.GRID_ADAPTED and not FORCE:
            print "Already set to grid adapted"
            return
        try:
            self._detector_ = cv2.GridAdaptedFeatureDetector(self._detector_)
        except:
            print "Could not create grid adapted detector"
            self.GRID_ADAPTED = False
        else:
            self.GRID_ADAPTED = True
            self.set_num_features = self.set_grid_adapted_num_features
            self.get_num_features = self.get_grid_adapted_num_features
    
    def reinit(self):
        """
        Reinitialise the feature detector if necessary
        """
        self._detector_init_()
        if self.GRID_ADAPTED:
            self.make_grid_adapted(FORCE=True)
    
    def get_features(self, im, mask=None):
        """
        get_features(image, mask) -> (keypoints, descriptors)
        """
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
    
    def set_grid_adapted_num_features(self, num_features):
        params_list = self._detector_.getParams()
        assert (self.GRID_ADAPTED and 
            "maxTotalKeypoints" in params_list), "Cannot set number of features"
        self._detector_.setInt("maxTotalKeypoints", num_features)
    
    def get_grid_adapted_num_features(self):
        params_list = self._detector_.getParams()
        assert (self.GRID_ADAPTED and 
            "maxTotalKeypoints" in params_list), "Cannot get number of features"
        num_features = self._detector_.getInt("maxTotalKeypoints")
        return num_features
    
    def set_num_features(self, num_features):
        raise UnboundLocalError("Convert to grid adapted detector first using make_grid_adapted()")
    
    def get_num_features(self):
        raise UnboundLocalError("Convert to grid adapted detector first using make_grid_adapted()")
    

class Sift(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Sift, self).__init__("SIFT", "SIFT", *args, **kwargs)
    

class Surf(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Surf, self).__init__("SURF", "SURF", *args, **kwargs)
    
    def set_hessian_threshold(self, hessian_threshold):
        if self.GRID_ADAPTED:
            maxTotalKeypoints = self._detector_.getInt("maxTotalKeypoints")
        self._detector_init_()
        self._detector_.setDouble("hessianThreshold", hessian_threshold)
        if self.GRID_ADAPTED:
            self.make_grid_adapted(FORCE=True)
            self.set_grid_adapted_num_features(maxTotalKeypoints)
    
    def get_nOctaves(self):
        assert "nOctaves" in self._detector_.getParams(), "Cannot set nOctaves, is it GridAdapted?"
        return self._detector_.getInt("nOctaves")
    
    def set_nOctaves(self, nOctaves):
        assert "nOctaves" in self._detector_.getParams(), "Cannot set nOctaves, is it GridAdapted?"
        self._detector_.setInt("nOctaves", nOctaves)
    

class Mser(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Mser, self).__init__("MSER", "ORB", *args, **kwargs)
    
    
class Orb(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__("ORB", "ORB", *args, **kwargs)
        self.set_num_features(1000)
    
    def set_num_features(self, num_features):
        self._detector_.setInt("nFeatures", np.int(num_features))
    
    def get_num_features(self):
        num_features = self._detector_.getInt("nFeatures")
        return num_features
    
    def set_nLevels(self, nLevels):
        self._detector_.setInt("nLevels", nLevels)
    
    def get_nLevels(self):
        nLevels = self._detector_.getInt("nLevels")
        return nLevels
    
    def set_scaleFactor(self, scaleFactor):
        self._detector_.setDouble("scaleFactor", scaleFactor)
    
    def get_scaleFactor(self):
        scaleFactor = self._detector_.getDouble("scaleFactor")
        return scaleFactor
    

class Freak(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Freak, self).__init__("FAST", "FREAK", *args, **kwargs)
    

class StableOrb(Orb):
    def __init__(self, *args, **kwargs):
        super(StableOrb, self).__init__(*args, **kwargs)
        self._detector_.setInt("nLevels", 1)
        self.nLevels = 3
        self.scaleFactor = 1.2
        self.ratio_threshold = 0.8
        self.flannmatcher = FlannMatcher(self.DESCRIPTOR_IS_BINARY)
    
    def get_nLevels(self):
        return self.nLevels
    
    def set_nLevels(self, nLevels):
        self.nLevels = nLevels
    
    def get_scaleFactor(self):
        return self.scaleFactor
    
    def set_scaleFactor(self, scaleFactor):
        self.scaleFactor = scaleFactor
    
    def get_ratio_threshold(self):
        return self.ratio_threshold
    
    def set_ratio_threshold(self, ratio_threshold):
        self.ratio_threshold = ratio_threshold
    
    def get_features(self, im, mask=None):
        # lambda to extract points from keypoint structure
        pts_from_kp = lambda kp: np.array([_kp_.pt for _kp_ in kp])
        # Shape of source image
        src_size = np.array(im.shape)
        im = np.asarray(im, dtype=np.float)
        # Size of pyramid images - base to apex
        pyramid_scale_factor = np.power(self.scaleFactor, range(self.nLevels))
        pyramid_shape = (src_size/(pyramid_scale_factor[:, np.newaxis])).astype(np.int)
        pyramid_imgs = [np.empty(_shape_) for _shape_ in pyramid_shape]
        
        # List of corresponding keypoints/descriptors for the pyramid
        keypoints = []
        descriptors = []
        pts = []
        #code.interact(local=locals())
        for (_shape_, idx) in zip(pyramid_shape, range(self.nLevels+1)):
            this_image = cv2.resize(im, tuple(_shape_)[::-1], pyramid_imgs[idx])
            if mask is None:
                this_mask = None
            else:
                this_mask = cv2.resize(im, tuple(_shape_)[::-1])
            # Can speed this up by sticking it into the next for loop and using
            # a mask to search regions based on candidate keypoints
            try:
                this_kp, this_desc = self._detect_(this_image.astype(np.uint8), this_mask)
            except:
                print "error in _detect_"
                code.interact(local=locals())
            keypoints.append(this_kp)
            descriptors.append(this_desc)
            pts.append(pts_from_kp(this_kp)*pyramid_scale_factor[idx])
        # Top-level keypoints
        candidate_points = (pts[-1]).copy()
        candidate_keypoints = np.asarray(copy.deepcopy(keypoints[-1]))
        candidate_descriptors = np.asarray(copy.deepcopy(descriptors[-1]))
        # Iterate over the pyramid from nLevels-1 to bottom
        for idx in range(self.nLevels-2, -1, -1):
            if candidate_points.shape[0] == 0:
                print "No candidate points left"
                code.interact(local=locals())
                break
            test_kp = keypoints[idx]
            test_desc = descriptors[idx]
            #test_pts = pts[idx]
            # Detect and match points between the levels
            print "idx = ", idx
            dam_result = self.flannmatcher.detect_and_match(
                candidate_keypoints, candidate_descriptors,
                test_kp, test_desc, self.ratio_threshold)
            c_valid_idx, t_valid_idx = dam_result[2:4]
            # Ignore points which are not detected at lower pyramid-levels
            px_difference = np.max(np.abs(candidate_points[c_valid_idx]-pts[idx][t_valid_idx]), axis=1)
            valid_px_difference = px_difference < 8.0
            select_idx = c_valid_idx[valid_px_difference]
            candidate_points = candidate_points[select_idx]
            candidate_keypoints = candidate_keypoints[select_idx]
            candidate_descriptors = candidate_descriptors[select_idx]
            
        self._keypoints_ = candidate_keypoints
        self._descriptors_ = candidate_descriptors
        return (self._keypoints_, self._descriptors_)
    

sift  = Sift
surf  = Surf
mser  = Mser
orb   = Orb
freak = Freak
