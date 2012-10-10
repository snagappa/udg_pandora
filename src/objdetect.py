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
from scipy import weave

#from matplotlib import pyplot
#from lib.common.misctools import cartesian_to_spherical as c2s

# Define services to enable/disable panel detection, valve detection
# (use an internal state?) and chain detection

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def uwsim_cam_calib(res_x=640, res_y=480):
    # See Section 4.2 of "Programming Computer Vision with Python" by 
    # Jan Erik Solem
    focal_x = 572*res_x/640
    focal_y = 356*res_y/480
    camera_matrix = np.diag([focal_x, focal_y, 1])
    camera_matrix[0, 2] = 0.5*res_x
    camera_matrix[1, 2] = 0.5*res_y
    return camera_matrix.astype(np.float)

def uwsim_stereo_cam_calib(res_x=640, res_y=480):
    camera_matrix = uwsim_cam_calib(res_x, res_y)
    camera_matrix = np.repeat(camera_matrix[np.newaxis], 2, 0)
    return camera_matrix

def uwsim_stereo_projection(res_x=640, res_y=480):
    stereo_camera_matrix = uwsim_stereo_cam_calib(res_x, res_y)
    r_t_matrix = np.repeat(np.eye(3, 4)[np.newaxis], 2, 0)
    r_t_matrix[1, 0, -1] = 0.1438
    projection_matrix = np.zeros((2, 3, 4))
    projection_matrix[0] = np.dot(stereo_camera_matrix[0], r_t_matrix[0])
    projection_matrix[1] = np.dot(stereo_camera_matrix[1], r_t_matrix[1])
    return projection_matrix
    
    
class STRUCT(object): pass

class FlannMatcher(object):
    """
    Wrapper class for using the Flann matcher. Attempts to use the new 
    FlannBasedMatcher interface, but uses the fallback flann_Index if this is
    unavailable.
    """
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS.copy()
        self.NEW_FLANN_MATCHER = False
        try:
            # Use the cv2 FlannBasedMatcher if available
            # bug : need to pass empty dict (#1329)        
            self._flann_ = cv2.FlannBasedMatcher(self.PARAMS, {})  
            self.NEW_FLANN_MATCHER = True
        except AttributeError, ae:
            rospy.loginfo(ae)
            rospy.loginfo(
                "Could not initialise FlannBasedMatcher, using fallback")
    
    def knnMatch(self, queryDescriptors, trainDescriptors, k=2, mask=None, 
                 compactResult=None):
        """
        knnMatch(queryDescriptors, trainDescriptors, k, mask=None, 
                 compactResult=None) -> idx1, idx2, distance
        Returns k best matches between queryDescriptors indexed by idx1 and 
        trainDescriptors indexed by idx2. Distance between the descriptors is
        given by distance, a Nxk ndarray.
        """
        if self.NEW_FLANN_MATCHER:
            matches = self._flann_.knnMatch(queryDescriptors, 
                                            trainDescriptors, k) #2
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
                        (this_match_dist, 
                         this_match_idx) = zip(*this_match_dist_idx)
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
            # bug: need to provide empty dict for params
            idx2, distance = self._flann_.knnSearch(queryDescriptors, k, 
                                                    params={})
        idx1 = np.arange(len(queryDescriptors))
        return idx1, idx2, distance
        

class Detector(object):
    def __init__(self, template=None, corners_3d=None, 
                 camera_matrix=uwsim_cam_calib(),
                 feat_detector=image_feature_extractor.orb, 
                 detector_preset=image_feature_extractor.DETECTOR_PRESET.MED):
        self._object_ = STRUCT()
        self._object_.template = None
        # Corners of the 3d object
        self._object_.corners_3d = None
        # Corners of the object in the template image
        self._object_.corners_2d = np.empty(0)
        # list of keypoints and descriptors of the template object
        self._object_.keypoints = None
        self._object_.descriptors = None
        # Homography matrix
        self._object_.h_mat = None
        # Return status of the FindHomography function
        self._object_.status = None
        # Number of inliers from homography
        self._object_.inliers = 0
        # Specify a threshold to binarise the test scene
        self._object_.intensity_threshold = None
        # Minimum required correspondences to validate localisation
        self._object_.min_correspondences = 10
        # Camera matrix
        self.camera_matrix = camera_matrix.astype(np.float)
        # Rotation matrix and translation vector from solvePnP
        self.r_mat = None
        self.t_vec = None
        # Image of the current scene
        self._scene_ = None
        # Create the feature detector
        self._detector_ = feat_detector(detector_preset)
        # Set up FLANN matcher
        self._flann_ = STRUCT()
        self._flann_.r_threshold = 0.6
        
        # Set up the parameters for FLANN matching
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
        # Initialise the flann matcher
        self._flann_.matcher = FlannMatcher(self._flann_.PARAMS)
        # Initialise the template
        self.set_template(template, corners_3d)
        
    
    def set_template(self, template_im=None, corners_3d=None):
        """
        Define a new template and corners of the object in the world.
        """
        # Detect features using the FINE preset
        if not template_im is None:
            template_im = self._binarise_(template_im, 
                                          self._object_.intensity_threshold)
            self._object_.template = template_im
            template_detector = self._detector_.__class__(
                image_feature_extractor.DETECTOR_PRESET.FINE)
            (self._object_.keypoints, self._object_.descriptors) = \
                template_detector.get_features(self._object_.template)
            height, width = self._object_.template.shape[:2]
            self._object_.corners_2d = np.float32([[0, 0], 
                                                   [width, 0], 
                                                   [width, height], 
                                                   [0, height]])
        if not corners_3d is None:
            assert_err_msg = "corners must a Nx3 ndarray, N>=4"
            assert type(corners_3d) is np.ndarray, assert_err_msg
            assert (corners_3d.ndim==2) and \
                   (corners_3d.shape[1]==3) and \
                   (corners_3d.shape[0]>3), assert_err_msg
            self._object_.corners_3d = corners_3d
        
    def _binarise_(self, im, intensity_threshold):
        bin_im = im.copy()
        if not intensity_threshold is None:
            bin_im[bin_im > intensity_threshold] = 255
            bin_im[bin_im <= intensity_threshold] = 0
        return bin_im
    
    def _detect_and_match_(self, obj_kp, obj_desc, scene_kp, scene_desc, 
                           ratio=0.75):
        """
        _detect_and_match_(obj_kp, obj_desc, scene_kp, scene_desc, ratio)
        Returns pt1, pt2, valid_idx1, valid_idx2
        """
        idx1, idx2, distance = self._flann_.matcher.knnMatch(obj_desc, 
                                                             scene_desc, 2)
        # Use only good matches
        mask = distance[:, 0] < (distance[:, 1] * ratio)
        mask[idx2[:, 1] == -1] = False
        valid_idx1 = idx1[mask]
        valid_idx2 = idx2[mask, 0]
        match_kp1, match_kp2 = [], []
        for (_idx1_, _idx2_) in zip(valid_idx1, valid_idx2):
            match_kp1.append(obj_kp[_idx1_])
            match_kp2.append(scene_kp[_idx2_])
        pts_1 = np.float32([kp.pt for kp in match_kp1])
        pts_2 = np.float32([kp.pt for kp in match_kp2])
        #kp_pairs = zip(match_kp1, match_kp2)
        return pts_1, pts_2, valid_idx1, valid_idx2 #, kp_pairs
    
    def detect(self, im_scene):
        """
        Calculates the homography matrix using the template and current scene.
        """
        if self._object_.template is None:
            print "object template is not set!"
            return None
        self._scene_ = copy.copy(im_scene)
        # Threshold the image
        self._scene_ = self._binarise_(self._scene_, 
                                       self._object_.intensity_threshold)
        (keypoints_scene, descriptors_scene) = (
            self._detector_.get_features(self._scene_))
        if not keypoints_scene:
            h_mat, status, inliers = None, False, 0
        else:
            dam_result = self._detect_and_match_(self._object_.keypoints,
                                                 self._object_.descriptors,
                                                 keypoints_scene,
                                                 descriptors_scene,
                                                 self._flann_.r_threshold)
            pts_1, pts_2 = dam_result[0:2]
            # Compute homography only if we have at least minimum required 
            # matches
            if len(pts_1) >= self._object_.min_correspondences:
                h_mat, status = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, 5.0)
                inliers = np.sum(status)
                print '%d / %d  inliers/matched' % (inliers, len(status))
                # Homography is valid only if we have at least 
                # min_correspondences number of inliers
                if ((h_mat is None) or 
                    (inliers < self._object_.min_correspondences)):
                    #h_mat, status, inliers = None, False, 0
                    status = False
                else:
                    status = True
            else:
                h_mat, status, inliers = None, False, 0
                #print '%d matches found,\
                #   not enough for homography estimation' % len(p1)
        self._object_.h_mat = h_mat
        self._object_.status = status
        self._object_.inliers = inliers
        #return h_mat, status
        
    def homography(self):
        """
        Return the estimated homography matrix
        """
        return self._object_.h_mat
        
    def _location_(self, corners_tpl_2d, corners_tpl_3d, h_mat, camera_matrix):
        # Compute the location if object corners are known in world coordinates
        retval = False
        r_mat = np.zeros((3, 3))
        tvec = np.zeros(3)
        if not corners_tpl_3d is None:
            # Project the corners of the template onto the scene
            corners_float = cv2.perspectiveTransform(
                corners_tpl_2d.reshape(1, -1, 2), h_mat).reshape(-1, 2)
            #corners = corners_float.astype(np.int32)
            #cv2.polylines(self._scene_, [corners], 
            #              True, (255, 255, 255), 4)
        
            retval, rvec, tvec = cv2.solvePnP(corners_tpl_3d, corners_float, 
                                              camera_matrix, np.empty(0))
            r_mat = cv2.Rodrigues(rvec)[0]
        return retval, r_mat, np.squeeze(tvec)
        # From http://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # "T is not the position of the camera. It is the position of 
        # the origin of the world coordinate system expressed in 
        # coordinates of the camera-centered coordinate system. The 
        # position, C, of the camera expressed in world coordinates is 
        # C = -R^{-1}T = -R^T T (since R is a rotation matrix)."
        #camera_centre = np.dot(-(cv2.Rodrigues(rvec)[0].T), tvec)
            
        
    def location(self):
        """
        Return detected (bool) and relative position (x, y, z) of the object.
        """
        detected = False
        position = np.zeros(3)
        if self._object_.status:
            (retval, self.r_mat, 
             self.t_vec) =  self._location_(self._object_.corners_2d,
                                            self._object_.corners_3d,
                                            self._object_.h_mat,
                                            self.camera_matrix)
            
            #sph_tvec = (c2s(tvec).T*[1, 180/np.pi, 180/np.pi])[0]
            obj_range = np.linalg.norm(self.t_vec)
            if not retval or (not 0.25 < obj_range < 5):
                print "Too close/far for reliable estimation"
            else:
                detected = True
                position = self.t_vec
            print "Relative panel position (x,y,z) = ", str(self.t_vec)
            #print "Relative panel position (r,az,el) = ", str(sph_tvec)
        return detected, position
            
    #def show(self):
    #    cv2.namedWindow("panel-detect")
    #    if not self._object_.h_mat is None:
    #        print "Detected the panel!"
    #    #else:
    #    #    print "No detection."
    #    if not self._scene_ is None:
    #        cv2.imshow("panel-detect", self._scene_)
            

class Stereo_detector(Detector):
    def __init__(self, template=None, corners_3d=None, 
                 camera_matrix=uwsim_stereo_cam_calib(),
                 camera_projection_matrix=uwsim_stereo_projection(),
                 feat_detector=image_feature_extractor.orb):
        assert ((type(camera_matrix) == np.ndarray) and
                (camera_matrix.shape == (2, 3, 3))), (
            "camera matrix must be a 2x3x3 ndarray")
        assert ((type(camera_projection_matrix) == np.ndarray) and
                (camera_projection_matrix.shape == (2, 3, 4))), (
            "camera matrix must be a 2x3x4 ndarray")
        # Initialise the detector using the parent class constructor
        super(Stereo_detector, self).__init__(None, None, 
            camera_matrix, feat_detector)
        # Additional parameters for the stereo detection
        self.camera_projection_matrix = (
            camera_projection_matrix.astype(np.float))
        # Additional homography for second camera
        self.template_z_offset = 0
        self._object_.keypoints_2d = None
        self._object_.keypoints_3d = None
        self._object_.affine_3d = None
        self.set_template(template, corners_3d)
        
    def set_template(self, template_im=None, corners_3d=None):
        super(Stereo_detector, self).set_template(template_im, corners_3d)
        if not self._object_.corners_3d is None:
            # Perform the homography and monocular localisation to obtain the
            # projection matrix as r_mat and t_vec
            super(Stereo_detector, self).detect(template_im)
            # parent location() requires that only one camera matrix is used
            _camera_matrix_ = self.camera_matrix
            self.camera_matrix = self.camera_matrix[0]
            (detected, panel_centre) = super(Stereo_detector, self).location()
            # Restore the camera matrix
            self.camera_matrix = _camera_matrix_
            # Save template z offset if required later
            self.template_z_offset = panel_centre[2]
            # Obtain pixel coordinates of the keypoints
            self._object_.keypoints_2d = np.array([_kp_.pt 
                for _kp_ in self._object_.keypoints])
            """
            # Append pixel coordinates with 1: (x, y) -> (x, y, 1)
            keypoints_2d_1 = np.hstack((self._object_.keypoints_2d, 
                np.ones((self._object_.keypoints_2d.shape[0], 1))))
            # Inverse mapping from pixel points to 3D world coordinates for
            # all the template keypoints
            code.interact(local=locals())
            self._object_.keypoints_3d = np.dot(
                np.dot(-self.r_mat.T, self.t_vec), keypoints_2d_1)
            """
            keypoints_3d = np.hstack((self._object_.keypoints_2d, 
                np.ones((self._object_.keypoints_2d.shape[0], 1))))
            kp_offset = np.array(self._object_.template.shape)/2
            keypoints_3d[:, 2] = self.template_z_offset
            keypoints_3d[:, 0:2] -= kp_offset
            keypoints_3d[:, 0:2] /= kp_offset
            keypoints_3d[:, 0:2] *= np.abs(self._object_.corners_3d[0, 0:2])
            self._object_.keypoints_3d = keypoints_3d
            
    
    def detect(self, im_scene_l, im_scene_r):
        if self._object_.template is None:
            print "object template is not set!"
            return None
        self._scene_ = [copy.copy(im_scene_l), copy.copy(im_scene_r)]
        # Threshold the image
        #intensity_threshold = self._object_.scene_intensity_threshold
        #self._scene_[self._scene_ > intensity_threshold] = 255
        #self._scene_[self._scene_ <= intensity_threshold] = 0
        
        # Set affine transformation and status to false
        self._object_.h_mat = [None, None]
        self._object_.affine_3d = [None, None]
        self._object_.status = np.zeros(2)
        self._object_.inliers = np.zeros(2)
        
        # Get keypoints and descriptors from the stereo image
        kp_l, desc_l = self._detector_.get_features(im_scene_l)
        kp_r, desc_r = self._detector_.get_features(im_scene_r)
        # Check if keypoints in both images, otherwise cannot do the matching
        if kp_l and kp_r:
            # Match keypoints in the images
            dam_result = self._detect_and_match_(kp_l, desc_l,
                                                 kp_r, desc_r,
                                                 self._flann_.r_threshold)
            pt_l, pt_r, idx_l, idx_r = dam_result[0:4]
            # Check for initial matches
            if idx_l.shape[0] < self._object_.min_correspondences:
                return
            # Valid matches are those where y co-ordinate of p1 and p2 are
            # almost equal
            y_diff = np.abs(pt_l[:, 1] - pt_r[:, 1])
            valid_pts_mask = y_diff < 3
            pt_l = pt_l[valid_pts_mask]
            pt_r = pt_r[valid_pts_mask]
            idx_l = idx_l[valid_pts_mask]
            idx_r = idx_r[valid_pts_mask]
            print "Matched ", str(len(valid_pts_mask)), " points between l & r images"
            
            # Proceed only if minimum required points detected
            if ((pt_l.shape[0] < self._object_.min_correspondences) or
                (pt_r.shape[0] < self._object_.min_correspondences)):
                return
            
            final_matched_kp_img = []
            final_matched_kp_tpl = []
            for kp_img, desc_img, idx_img in ((kp_l, desc_l, idx_l), 
                                              (kp_r, desc_r, idx_r)):
                # Match template descriptors with valid points from left image
                selected_keypoints_img = [kp_img[_idx_] for _idx_ in idx_img]
                selected_descriptors_img = desc_img[idx_img]
                
                dam_result = self._detect_and_match_(self._object_.keypoints,
                                                     self._object_.descriptors,
                                                     selected_keypoints_img,
                                                     selected_descriptors_img)
                template_idx = dam_result[2]
                if template_idx.shape[0] < self._object_.min_correspondences:
                    return
                final_matched_kp_tpl.append(dam_result[0])
                final_matched_kp_img.append(dam_result[1])
            
            # Matching keypoints are now stored in
            # final_matched_kp_{tpl,img}[0->left, 1->right]
            h_mat = []
            status = np.zeros(2)
            inliers = np.zeros(2)
            for count in range(2):
                # Compute homography since we have minimum required matches
                _h_mat_, _status_ = cv2.findHomography(final_matched_kp_tpl[count],
                                                   final_matched_kp_img[count],
                                                   cv2.RANSAC, 5.0)
                _inliers_ = np.sum(_status_)
                print '%d / %d  inliers/matched' % (_inliers_, len(_status_))
                
                inliers[count] = _inliers_
                # Homography is valid only if we have at least 
                # min_correspondences number of inliers
                if ((_h_mat_ is None) or 
                    (_inliers_ < self._object_.min_correspondences)):
                    status[count] = False
                else:
                    status[count] = True
                h_mat.append(_h_mat_)
                
            self._object_.h_mat = h_mat
            self._object_.status = status
            self._object_.inliers = inliers
            """
            # Triangulate the points
            points4d = cv2.triangulatePoints(self.camera_projection_matrix[0],
                                             self.camera_projection_matrix[1],
                                             pt_l.T, pt_r.T)
            points4d /= points4d[3]  #(wx,wy,wz,w) -> (x,y,z,1)
            
            
            # Select the subset of triangulated points using the valid matches
            # of the template
            template_idx, scene_idx = dam_result[2:4]
            scene_points = points4d[0:3, scene_idx]
            template_points = self._object_.keypoints_3d[template_idx]
            print "Matched ", str(len(template_idx)), " points between l image and template"
            # Use the point correspondences to obtain the 3D affine 
            # transformation
            # Python binding for estimateAffine3D doesn't work, use C api
            #(retval, affine_3d, inliers) = (
            #    cv2.estimateAffine3D(template_points.astype(np.float32), 
            #                         scene_points.T))
            
            # Check if sufficient number of correspondences
            if template_idx.shape[0] < self._object_.min_correspondences:
                return
            (retval, affine_3d, inliers) = (
                _estimateAffine3D_(template_points, scene_points.T))
            print "estimateAffine3D found", str(inliers[0]), " inliers"
            # Find the transformation of [0, 0, 0, 1] using the transformation
            if retval and len(inliers) >= self._object_.min_correspondences:
                self._object_.affine_3d = affine_3d
                self._object_.status = True
            """
    
    """
    def affine_3d(self):
        # Return the estimated 3D affine transformation
        return self._object_.affine_3d
    
    def location(self):
        # Return detected (bool) and relative position (x, y, z) of the object.
        detected = False
        position = np.zeros(3)
        # Evaluate the location only if the homography is valid
        if self._object_.status:
            detected = True
            position = np.dot(self._object_.affine_3d, 
                              np.array([[0, 0, 0, 1]]).T)
        return detected, position
    """
    
    def location(self):
        """
        Return detected (bool) and relative position (x, y, z) of the object.
        """
        detected = 0
        position = np.zeros(3)
        
        r_mat = []
        t_vec = []
        for count in range(2):
            # Compute the location if a homography was computed even if invalid
            if not self._object_.h_mat[count] is None:
                (_retval_, _r_mat_, 
                 _t_vec_) = self._location_(self._object_.corners_2d,
                                            self._object_.corners_3d,
                                            self._object_.h_mat[count],
                                            self.camera_matrix[count])
                r_mat.append(_r_mat_)
                t_vec.append(_t_vec_)
                #sph_tvec = (c2s(tvec).T*[1, 180/np.pi, 180/np.pi])[0]
                obj_range = np.linalg.norm(_t_vec_)
                # Check if the computed location is valid
                if ((_retval_) and (0.25 < obj_range < 5) and 
                    (self._object_.status[count])):
                    detected += 1
                    position += _t_vec_
                else:
                    print "Too close/far for reliable estimation"
            else:
                # Cannot compute homography, push None on to the lists
                r_mat.append(None)
                t_vec.append(None)
            
            if detected:
                position /= detected
                detected = True
                print "Relative panel position (x,y,z) = ", str(position)
            self.r_mat = r_mat
            self.t_vec = t_vec
            
            #print "Relative panel position (r,az,el) = ", str(sph_tvec)
        return detected, position

def _estimateAffine3D_(src_points, dst_points):
    """
    (retval, affine_3d, inliers) = _estimateAffine3D_(np_pts_from, np_pts_to)
    """
    err_msg = "points must be a Nx3 ndarray"
    assert np.ndim(src_points) == np.ndim(dst_points) == 2, err_msg
    assert src_points.shape[1] == dst_points.shape[1] == 3, err_msg
    assert ((src_points.shape[0] > 0) and
            (dst_points.shape[0] > 0)), err_msg
    if not src_points.dtype == np.float32:
        print "src_points should have dtype=np.float32, converting..."
        np_points_from = src_points.astype(np.float32)
    else:
        np_points_from = src_points
    if not dst_points.dtype == np.float32:
        print "dst_points should have dtype=np.float32, converting..."
        np_points_to = dst_points.astype(np.float32)
    else:
        np_points_to = dst_points
    
    affine_3d = np.zeros((3, 4), dtype=np.float64)
    inliers = []
    num_inliers = 0
    retval = 0
    
    header_code = """
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv/cvaux.h>
#include <string>
#include <iostream>
"""
    fn_code = """
int num_points_from, num_points_to, result;

// Number of points is the number of rows
num_points_from = Nnp_points_from[0];
num_points_to = Nnp_points_to[0];

// Create affine matrix
cv::Mat aff(3, 4, CV_64F, affine_3d);

// vector of inliers
std::vector<uchar> inliers;

// Prepare two sets of 3D points
cv::Mat mat_points_from(num_points_from, 3, CV_32F, np_points_from);
cv::Mat mat_points_to(num_points_to, 3, CV_32F, np_points_to);
// Convert the Mat to vectors of Point3f, otherwise RANSAC will fail
std::vector<cv::Point3f> points_from = cv::Mat_<cv::Point3f>(mat_points_from);
std::vector<cv::Point3f> points_to = cv::Mat_<cv::Point3f>(mat_points_to);

result = cv::estimateAffine3D(points_from, points_to, aff, inliers);
num_inliers = inliers.size();
retval = result;
"""
    ros_root = "/opt/ros/fuerte/"
    python_vars = ["np_points_from", "np_points_to", "affine_3d", 
                   "retval", "num_inliers"]
    weave.inline(fn_code, python_vars, support_code=header_code,
                 libraries=["opencv_calib3d"],
                 library_dirs=[ros_root+"lib"],
                 include_dirs=[ros_root+"include", ros_root+"include/opencv"],
                 extra_compile_args=["-O2 -g"], verbose=1)
    inliers = [num_inliers]
    return not(retval), affine_3d, inliers