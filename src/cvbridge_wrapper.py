# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:07:01 2012

@author: snagappa
"""

import roslib
roslib.load_manifest('udg_pandora')
import cv
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    def __init__(self):
        """image_converter() -> converter
        Create object to convert sensor_msgs.msg.Image type to cv_image
        """
        self.bridge = CvBridge()
        self._cv_image_ = None;
    
    def cvimage(self, data, COLOUR_FMT=None): #or set format to 'passthrough'?
        """cvimage(self, data, COLOUR_FMT=None) -> cv_image
        Convert data from ROS sensor_msgs Image type to OpenCV using the 
        specified colour format
        """
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
        """cvimage(self, data, COLOUR_FMT=None) -> cv_image
        Convert data from ROS sensor_msgs Image type to OpenCV with colour
        format "mono8"
        """
        return self.cvimage(data, 'mono8') # Use 'rgb8' instead?
    
    def img_msg(self, cvim, encoding="passthrough"):
        """img_msg(self, cvim, encoding="passthrough") -> imgmsg
        Convert OpenCV image to ROS sensor_msgs Image
        """
        return self.bridge.cv_to_imgmsg(cvim, encoding)