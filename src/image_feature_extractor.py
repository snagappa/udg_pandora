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

"""
SURF_PARAMS = collections.namedtuple("SURF_PARAMS", ['hessianThreshold', 
                                                     'nOctaves', 
                                                     'nOctaveLayers',
                                                     'extended',
                                                     'upright'])
ORB_PARAMS = collections.namedtuple("ORB_PARAMS", ['threshold'])
"""
class _feature_detector_(object):
    def __init__(self):
        self.PARAMS = None
        self._keypoints_ = None
        self._descriptors_ = None
        self._detector_ = lambda: 0
        self._detector_.detectAndCompute = lambda im, mask, *args, **kwargs: (None, None)
        self._detect_ = lambda im, mask: self._detector_.detectAndCompute(im, mask, 
            useProvidedKeypoints = False)
        
    def get_features(self, im, mask=None):
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
        

class surf(object):
    def __init__(self, PRESET=DETECTOR_PRESET.MED):
        hessianThreshold = 400
        nOctaves = 4
        nOctaveLayers = 2
        extended = 1
        upright = False
        if PRESET == DETECTOR_PRESET.FINE:
            hessianThreshold = 200
        elif PRESET == DETECTOR_PRESET.MED:
            hessianThreshold = 400
        else:
            hessianThreshold = 800
        #PARAMS = SURF_PARAMS(minHessian, nOctaves, nOctaveLayers, 
        #                     extended, upright)
        PARAMS = dict(hessianThreshold=hessianThreshold, 
                      nOctaves=nOctaves,
                      nOctaveLayers=nOctaveLayers,
                      extended=extended,
                      upright=upright)
        self.PARAMS = PARAMS
        self.NORM = cv2.NORM_L2
        self.DESCRIPTOR_IS_BINARY = False
        self._keypoints_ = None
        self._descriptors_ = None
        self._detector_ = cv2.SURF(*PARAMS)
        self._detect_ = lambda im, mask: self._detector_.detectAndCompute(im, mask, 
            useProvidedKeypoints = False)
        
    def reinit(self):
        self._detector_ = cv2.SURF(**self.PARAMS)
    
    def get_features(self, im, mask=None):
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
    

class orb(object):
    def __init__(self, PRESET=DETECTOR_PRESET.MED):
        if PRESET == DETECTOR_PRESET.FINE:
            nfeatures = 1000
            nlevels = 16
        elif PRESET == DETECTOR_PRESET.MED:
            nfeatures = 1000
            nlevels = 12
        else:
            nfeatures = 500
            nlevels = 8
        #PARAMS = ORB_PARAMS(threshold)
        PARAMS = dict(nfeatures=nfeatures, nlevels=nlevels)
        self.PARAMS = PARAMS
        self.NORM = cv2.NORM_HAMMING
        self.DESCRIPTOR_IS_BINARY = True
        self._keypoints_ = None
        self._descriptors_ = None
        self._detector_ = cv2.ORB(**PARAMS)
        self._detect_ = lambda im, mask: self._detector_.detectAndCompute(im, mask, 
            useProvidedKeypoints = False)
        
    def reinit(self):
        self._detector_ = cv2.SURF(**self.PARAMS)
    
    def get_features(self, im, mask=None):
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
    
    
#img = cv2.imread('messi4.jpg')

# Convert them to grayscale
#imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
