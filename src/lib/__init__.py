# -*- coding: utf-8 -*-
__all__ = ["blas", "misctools", "kalmanfilter",  "pointclouds", "pc2wrapper", "image_feature_extractor", "cameramodels"]

import blas
import misctools
import kalmanfilter

try:
    import pointclouds
except:
    print "Could not import pointclouds. Is ROS initialised?"
    
try:
    import pc2wrapper
except:
    print "Could not import pc2wrapper. Is ROS initialised?"
try:
    del __blas_c_code__
except:
    pass

import image_feature_extractor
import cameramodels
