# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:28:00 2012

@author: snagappa
"""
import roslib
roslib.load_manifest('udg_pandora')
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

    
class detector(object):
    def __init__(self, template=None, detector=image_feature_extractor.orb, detector_params=tuple()):
        self._object_ = STRUCT()
        self._object_.template = None
        self._object_.corners = np.empty(0)
        self._object_.keypoints = None
        self._object_.descriptors = None
        self._object_.H = None
        self._object_.status = None
        self._scene_ = None
        self._detector_ = detector(*detector_params)
        # Set up FLANN matcher
        self._flann_ = STRUCT()
        self._flann_.r_threshold = 0.6
        if self._detector_.NORM == cv2.NORM_L2:
            self._flann_.PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, 
                                       trees = 5)
        else:
            self._flann_.PARAMS = dict(algorithm = FLANN_INDEX_LSH,
                                       table_number = 6, # 12
                                       key_size = 12,     # 20
                                       multi_probe_level = 1) #2
        self._flann_.matcher = cv2.FlannBasedMatcher(self._flann_.PARAMS, {})  # bug : need to pass empty dict (#1329)        
        
        if not template is None:
            self.set_template(template.copy())
        
    
    def set_template(self, template_im):
        self._object_.template = template_im.copy()
        (self._object_.keypoints, self._object_.descriptors) = \
            self._detector_.get_features(self._object_.template)
        h1, w1 = self._object_.template.shape[:2]
        self._object_.corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            
    def _detect_and_match_(self, obj_kp, obj_desc, scene_kp, scene_desc, ratio=0.75):
        matches = self._flann_.matcher.knnMatch(obj_desc, 
                                                trainDescriptors=scene_desc, 
                                                k = 2) #2
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( obj_kp[m.queryIdx] )
                mkp2.append( scene_kp[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
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
            
            if len(p1) >= 30:
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
        self._object_status = status
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
            
        
        
        
        """
        self._flann_.flann = cv2.flann_Index(descriptors_scene, 
                                             self._flann_.params)
        k=2 # find the 2 nearest neighbors
        idx2, dists = self._flann_.flann.knnSearch(self._panel_.descriptors, 
            k, params = {}) # bug: need to provide empty dict
        mask = dists[:,0] / dists[:,1] < self._flann_.r_threshold
        idx1 = np.arange(len(descriptors_scene))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        valid_pairs = pairs[mask]
        """
        
        
        """
        // PROCESS NEAREST NEIGHBOR RESULTS
        // Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
        std::vector<cv::Point2f> mpts_1, mpts_2; // Used for homography
        std::vector<int> indexes_1, indexes_2; // Used for homography
        std::vector<uchar> outlier_mask;  // Used for homography
        for(int i=0; i<descriptors_object.rows; ++i)
        {
            // Check if this descriptor matches with those of the objects
            // Apply NNDR
            if(dists.at<float>(i,0) <= nndrRatio * dists.at<float>(i,1))
            {
                mpts_1.push_back(keypoints_object.at(i).pt);
                indexes_1.push_back(i);
    
                mpts_2.push_back(keypoints_scene.at(results.at<int>(i,0)).pt);
                indexes_2.push_back(results.at<int>(i,0));
            }
        }
    
        // FIND HOMOGRAPHY
        unsigned int minInliers = 8;
        if(mpts_1.size() >= minInliers)
        {
            cv::Mat H = findHomography(mpts_1, mpts_2, cv::RANSAC, 1.0, outlier_mask);
            int inliers=0, outliers=0;
            for(unsigned int k=0; k<mpts_1.size();++k)
            {
                if(outlier_mask.at(k))
                { ++inliers; }
                else
                { ++outliers; }
            }
            printf("Inliers=%d Outliers=%d\n", inliers, outliers);
            //-- Get the corners from the image_1 ( the object to be "detected" )
            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0);
            obj_corners[1] = cvPoint( img_object.cols, 0 );
            obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
            obj_corners[3] = cvPoint( 0, img_object.rows );
            std::vector<Point2f> scene_corners(4);
    
            perspectiveTransform( obj_corners, scene_corners, H);
            return(scene_corners);
    
        }
        else
        {
            printf("Not enough matches (%d) for homography...\n", (int)mpts_1.size());
            std::vector<Point2f> scene_corners(0);
            return(scene_corners);
        }
    """
