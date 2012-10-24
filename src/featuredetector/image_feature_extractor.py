# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:33:17 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('udg_pandora')
import cv2
#import collections

class DETECTOR_PRESET:
    FINE, MED, COARSE = range(3)

class STRUCT(object): pass

DETECTORS = ("SIFT", "SURF", "FAST", "MSER", "ORB")
BINARY_DESCRIPTORS = ("ORB", "FREAK")
DESCRIPTORS = ("SIFT", "SURF") + BINARY_DESCRIPTORS


class _feature_detector_(object):
    def __init__(self, feature_detector, feature_descriptor):
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
        
    
    def reinit(self):
        """
        Reinitialise the feature detector if necessary
        """
        self._detector_init_()
    
    def get_features(self, im, mask=None):
        """
        get_features(image, mask) -> (keypoints, descriptors)
        """
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
        
    def change_num_features(self, parameter_scale_factor):
        """
        change_num_features(parameter_scale_factor)
        Scale the parameter which affects the number of keypoints returned.
        Scale factor > 1 increases the number of keypoints, < 1 decreases the
        number of keypoints.
        """
        assert parameter_scale_factor > 0, "scale factor must be > 0"
        if self.type.feature_detector == "ORB":
            num_features = self._detector_.getInt("nFeatures")
            num_features *= parameter_scale_factor
            self._detector_.setInt("nFeatures", int(num_features))
        elif self.type.feature_detector == "SURF":
            hessian_threshold = self._detector_.getDouble("hessianThreshold")
            hessian_threshold /= parameter_scale_factor
            self._detector_.setDouble("hessianThreshold", float(hessian_threshold))


class Sift(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Sift, self).__init__("SIFT", "SIFT")
    
class Surf(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Surf, self).__init__("SURF", "SURF")
    
class Mser(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Mser, self).__init__("MSER", "ORB")
    
class Orb(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__("ORB", "ORB")
        self.change_num_features(2)
        
class Freak(_feature_detector_):
    def __init__(self, *args, **kwargs):
        super(Freak, self).__init__("FAST", "FREAK")

sift  = Sift
surf  = Surf
mser  = Mser
orb   = Orb
freak = Freak
