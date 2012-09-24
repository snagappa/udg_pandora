# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:33:17 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('udg_pandora')
import cv2
import collections

SURF_PARAMS = collections.namedtuple("SURF_PARAMS", ['hessianThreshold', 
                                                     'nOctaves', 
                                                     'nOctaveLayers',
                                                     'extended',
                                                     'upright'])
ORB_PARAMS = collections.namedtuple("ORB_PARAMS", ['threshold'])

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
    def __init__(self, minHessian=400):
        PARAMS = SURF_PARAMS(minHessian, 4, 2, 1, False)
        self.PARAMS = PARAMS
        self.NORM = cv2.NORM_L2
        self._keypoints_ = None
        self._descriptors_ = None
        self._detector_ = cv2.SURF(*PARAMS)
        self._detect_ = lambda im, mask: self._detector_.detectAndCompute(im, mask, 
            useProvidedKeypoints = False)
        
    def get_features(self, im, mask=None):
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
    

class orb(object):
    def __init__(self, threshold=10000):
        PARAMS = ORB_PARAMS(threshold)
        self.PARAMS = PARAMS
        self.NORM = cv2.NORM_HAMMING
        self._keypoints_ = None
        self._descriptors_ = None
        self._detector_ = cv2.ORB(*PARAMS)
        self._detect_ = lambda im, mask: self._detector_.detectAndCompute(im, mask, 
            useProvidedKeypoints = False)
        
    def get_features(self, im, mask=None):
        (self._keypoints_, self._descriptors_) = \
            self._detect_(im, mask)
        return (self._keypoints_, self._descriptors_)
    
    
#img = cv2.imread('messi4.jpg')

# Convert them to grayscale
#imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
