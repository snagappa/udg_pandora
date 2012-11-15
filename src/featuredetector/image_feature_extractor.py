# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:33:17 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('udg_pandora')
import cv2
from lib.common.misctools import STRUCT
#import collections

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
    
    def set_num_features(self, num_features):
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
    

class Mser(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Mser, self).__init__("MSER", "ORB", *args, **kwargs)
    
    
class Orb(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__("ORB", "ORB", *args, **kwargs)
        self.set_num_features(1000)
    
    def set_num_features(self, num_features):
        self._detector_.setInt("nFeatures", num_features)
    

class Freak(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Freak, self).__init__("FAST", "FREAK", *args, **kwargs)


sift  = Sift
surf  = Surf
mser  = Mser
orb   = Orb
freak = Freak
