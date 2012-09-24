# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:07:01 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('objdetect')
import cv
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self._cv_image_ = None;
    
    def cvimage(self, data, COLOUR_FMT=None): #or set format to 'passthrough'?
        try:
            if COLOUR_FMT is None:
                cv_image = self.bridge.imgmsg_to_cv(data)
            else:
                cv_image = self.bridge.imgmsg_to_cv(data, COLOUR_FMT)
        except CvBridgeError, e:
            self._cv_image_ = None
            print e
        self._cv_image_ = cv_image
        return self._cv_image_
        
    def cvimagegray(self, data):
        return self.cvimage(data, 'mono8') # Use 'rgb8' instead?
