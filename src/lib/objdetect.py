# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:28:00 2012

@author: snagappa
"""
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#import phasecorr
import cv2
import numpy as np
#import copy
from scipy import weave
from misctools import STRUCT, normalize_angle #, estimate_rigid_transform_3d
import cameramodels
import image_feature_extractor
import tf
import scipy.optimize
_minimize_ = getattr(scipy.optimize, "minimize", None)


#import np.linalg.LinAlgError as LinAlgError
from numpy.linalg.linalg import LinAlgError
import traceback
import sys

#from IPython import embed

#from matplotlib import pyplot
#from lib.common.misctools import cartesian_to_spherical as c2s

# Define services to enable/disable panel detection, valve detection
# (use an internal state?) and chain detection

class Detector(object):
    def __init__(self, feat_detector=image_feature_extractor.orb):
        self._object_ = STRUCT()
        self._object_.template = None
        self._object_.template_mask = None
        # Corners of the 3d object
        self._object_.corners_3d = None
        # Corners of the object in the template image
        self._object_.corners_2d = np.empty(0)
        # list of keypoints and descriptors of the template object
        self._object_.keypoints = None
        self._object_.descriptors = None
        # Return status of the FindHomography function
        self._object_.status = False
        # Homography matrix
        self._object_.h_mat = None
        # Number of inliers from homography
        self._object_.num_inliers = 0
        # Status of the inliers
        self._object_.inliers_status = []
        # Specify a threshold to binarise the test scene
        self._object_.intensity_threshold = None
        # Minimum required correspondences to validate localisation
        self._object_.min_correspondences = 10
        # Near and far distances for detection
        self._object_.near_distance = 0.25
        self._object_.far_distance = 5
        
        # Store matched points
        self._matches_ = STRUCT()
        # Keypoints from the template
        self._matches_.kp_obj = None
        # Matching keypoints in the scene
        self._matches_.kp_scn = None
        
        # Camera
        self.camera = cameramodels.PinholeCameraFeatureDetector(feat_detector)
        self._camera_params_ = STRUCT()
        self._camera_params_.camera_matrix = None
        self._camera_params_.kscale = 0
        self._camera_params_.kvec_scaled = None
        self._camera_params_.camera_cov_scaled = None
        self._camera_params_.inv_camera_matrix = None
        # Mapping from template to world
        self._camera_params_.wNm = None
        self._camera_params_.wNm_flat = None
        self._camera_params_.inv_wNm = None
        self._camera_params_.wNm_cov = None
        
        # Rotation matrix and translation vector
        self.obj_rpy = None
        self.obj_trans = None
        self.obj_cov = None
        self.obj_corners = np.empty(0)
        
        # Pose estimated by OpenCV
        self.opencv_obj_trans = None
        self.opencv_obj_rpy = None
        # Pose estimated using Sturm method
        self.sturm_obj_trans = None
        self.sturm_obj_rpy = None
        
        # Image of the current scene
        self._scene_ = None
        # Set up FLANN matcher
        self.flann_ratio_threshold = 0.75
        
        # Filter for the template and scene
        #self.filter_kernel = phasecorr.log_filter_kernel(13)
    
    def init_camera(self, camera_info, *dummy_args):
        self.camera.fromCameraInfo(camera_info)
        camera_matrix = (self.camera.projection_matrix()[:, :3]).copy()
        # Normalise(?) camera matrix
        camera_matrix /= camera_matrix[2, 2]
        # Normalisation of camera matrix covariance (use Frobenius norm)
        kscale = 1./np.linalg.norm(camera_matrix) 
        self._camera_params_.camera_matrix = camera_matrix
        self._camera_params_.kscale = kscale
        self._camera_params_.kvec_scaled = (
            camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]] * kscale)
        camera_cov = np.eye(4)
        self._camera_params_.camera_cov_scaled = camera_cov * kscale**2
        self._camera_params_.inv_camera_matrix = np.linalg.inv(camera_matrix)
    
    def get_wNm(self):
        return self._object_.wNm.copy()
    
    def set_wNm(self, wNm, wNm_cov=None, FORCE=False):
        wNm_norm_test = np.abs(np.linalg.norm(wNm) - 1) < 1e-6
        if not FORCE:
            assert wNm_norm_test, "Norm of wNm must be 1"
        elif not wNm_norm_test:
            print "Norm of wNm not equal 1, but FORCE=True"
        self._camera_params_.wNm = wNm.copy()
        self._camera_params_.wNm_flat = wNm.flatten()
        self._camera_params_.inv_wNm = np.linalg.inv(wNm)
        if wNm_cov is None:
            self._camera_params_.wNm_cov = np.zeros((9, 9))
        else:
            self._camera_params_.wNm_cov = wNm_cov.copy()
    
    def set_flann_ratio_threshold(self, flann_ratio_threshold):
        self.flann_ratio_threshold = flann_ratio_threshold
    
    def set_near_far(self, near=None, far=None):
        if not near is None:
            self._object_.near_distance = near
        if not far is None:
            self._object_.far_distance = far
    
    def set_template(self, template_im=None, corners_3d=None, template_mask=None):
        """
        Define a new template and corners of the object in the world.
        """
        # Detect features using the FINE preset
        if not template_im is None:
            template_im = self.process_images((template_im,))[0]
            self._object_.template = template_im
            self._object_.template_mask = template_mask
            (self._object_.keypoints, self._object_.descriptors) = \
                self.camera.get_features(self._object_.template,
                                         self._object_.template_mask)
            height, width = self._object_.template.shape[:2]
            self._object_.corners_2d = np.float32([[0, 0], 
                                                   [width, 0], 
                                                   [width, height], 
                                                   [0, height]])
        self.set_corners3d(corners_3d)
    
    def set_corners3d(self, corners_3d):
        if not corners_3d is None:
            assert_err_msg = "corners must a Nx3 ndarray, N>=4"
            assert type(corners_3d) is np.ndarray, assert_err_msg
            assert (corners_3d.ndim==2) and \
                   (corners_3d.shape[1]==3) and \
                   (corners_3d.shape[0]>3), assert_err_msg
            self._object_.corners_3d = corners_3d
    
    def add_to_template(self, template_im=None, template_mask=None):
        if template_im is None or self._object_.corners_3d is None:
            print "Unable to add to template!"
            return
        if self._object_.template is None:
            self.set_template(template_im, template_mask=template_mask)
        else:
            template_im = self.process_images((template_im,))[0]
            orig_tpl_shape = self._object_.template.shape
            #extra_tpl_shape = np.asarray(template_im.shape)
            #scalefactor = orig_tpl_shape/extra_tpl_shape
            template_im = cv2.resize(template_im, orig_tpl_shape[::-1])
            template_mask = cv2.resize(template_mask, orig_tpl_shape[::-1])
            (extra_keypoints, extra_descriptors) = \
                self.camera.get_features(template_im, template_mask)
            if extra_descriptors is None or extra_descriptors.shape[0] == 0:
                print "No keypoints in extra template"
                return
            #for _kp_ in extra_keypoints:
            #    _kp_.pt = tuple(np.asarray(_kp_.pt)*scalefactor)
            self._object_.keypoints += extra_keypoints
            self._object_.descriptors = np.vstack((self._object_.descriptors, extra_descriptors))
    
    def process_images(self, images=()):
        assert type(images) is tuple, "Images must be a tuple"
        # If images are colour, convert to grayscale, otherwise return as is
        return [self._gray_(_im_) for _im_ in images]
        #return [self._binarise_(cv2.filter2D(_im_, -1, self.log_kernel),
        #                        self._object_.intensity_threshold)
        #        for _im_ in images]
        #return [self._sharpen_(_im_) for _im_ in images]
        #return images
    
    def _gray_(self, im):
        # Check if more than one channel
        if (im.ndim == 2) or ((im.ndim == 3) and (im.shape[2] == 1)):
            return im.copy()
        else:
            return cv2.cvtColor(im, cv2.cv.CV_BGR2GRAY)
    
    def _binarise_(self, im, intensity_threshold):
        if not intensity_threshold is None:
            im[im > intensity_threshold] = 255
            im[im <= intensity_threshold] = 0
        return im
    
    def _sharpen_(self, im, filter_size=(5, 5), alpha=1.5, beta=-0.5):
        sm_im = cv2.GaussianBlur(im, filter_size, 0)
        return cv2.addWeighted(im, alpha, sm_im, beta, 1.0)
    
    def _log_filter_(self, im):
        log_im = cv2.filter2D(im, -1, self.filter_kernel)
        return log_im
    
    def _detect_(self, im_scene):
        if self._object_.template is None:
            print "object template is not set!"
            return None
        #self._scene_ = copy.copy(im_scene)
        # Filter the scene
        proc_im_scene = self.process_images((im_scene,))[0]
        (keypoints_scene, descriptors_scene) = (
            self.camera.get_features(proc_im_scene))
        #for _kp_ in keypoints_scene:
        #    self._scene_[_kp_.pt[1], _kp_.pt[0]] = 255
        if not descriptors_scene is None and descriptors_scene.shape[0] > 2:
            dam_result = self.camera.detect_and_match(self._object_.keypoints,
                                                      self._object_.descriptors,
                                                      keypoints_scene,
                                                      descriptors_scene,
                                                      self.flann_ratio_threshold)
            pts_obj, pts_scn = dam_result[0:2]
            status, h_mat, num_inliers, inliers_status = (
                self.camera.find_homography(pts_obj, pts_scn, ransacReprojThreshold=1,
                                            min_inliers=self._object_.min_correspondences))
        else:
            pts_scn = np.zeros(0)
            pts_obj = np.zeros(0)
            status = False
            inliers_status = np.zeros(0, dtype=np.bool)
            num_inliers = 0
            h_mat = None
        
        # Save only the inliers
        inliers_idx = inliers_status.astype(np.bool)
        self._object_.pts_obj = pts_obj[inliers_idx]
        self._object_.pts_scn = pts_scn[inliers_idx]
        #try:
        #    # Compute the inverse if findHomography(pts_scn, pts_obj)
        #    h_mat = np.linalg.inv(h_mat)
        #except:
        #    print "Error computing inverse of homography!"
        #    h_mat = None
        #    status = False
        #    #self._matches_.kp_obj = None
        #    #self._matches_.kp_scn = None
        return status, h_mat, num_inliers, inliers_status
    
    def detect(self, im_scene, *dummy_args):
        """
        detect(self, im_scene) -> None
        Detects object by matching features and calculating the homography
        matrix from the template and current scene.
        """
        self._scene_ = im_scene
        status, h_mat, num_inliers, inliers_status = self._detect_(im_scene)
        print '%d / %d  inliers/matched' % (num_inliers, len(inliers_status))
        self._object_.status = status
        self._object_.h_mat = h_mat
        self._object_.num_inliers = num_inliers
        self._object_.inliers_status = inliers_status
        #if not h_mat is None and len(h_mat):
        #    print "|H| = ", np.linalg.det(h_mat)
        return status
    
    def homography(self):
        """
        Return the estimated homography matrix
        """
        return self._object_.h_mat
    
    def location(self, USE_STURM=False, CROSS_VERIFY=False,
                 VERIFY_MAX_ERR_M=0.05, VERIFY_MAX_ERR_RAD=0.035):
        """
        Return detected (bool), relative position (x, y, z) and
        orientation (RPY) of the object.
        """
        # From http://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # "T is not the position of the camera. It is the position of
        # the origin of the world coordinate system expressed in
        # coordinates of the camera-centered coordinate system. The
        # position, C, of the camera expressed in world coordinates is
        # C = -R^{-1}T = -R^T T (since R is a rotation matrix)."
        #camera_centre = np.dot(-(cv2.Rodrigues(rvec)[0].T), tvec)
        detected = False
        camera_params = self._camera_params_
        if self._object_.status:
            # Project the corners of the template onto the scene
            self.obj_corners = cv2.perspectiveTransform(
                self._object_.corners_2d.reshape(1, -1, 2),
                self._object_.h_mat).reshape(-1, 2)
            # Cannot use Sturm method if wNm not specified.
            # Don't cross-verify Sturm with OpenCV unless
            # USE_STURM=True and CROSS_VERIFY=True
            if camera_params.wNm is None or not USE_STURM:
                USE_STURM = False
                CROSS_VERIFY = False
            USE_OPENCV = not USE_STURM or CROSS_VERIFY
            
            if USE_STURM:
                # Use Sturm method to generate pose and optimize if possible
                # Mimini should be the same as h_mat
                #Mimini, _CONDS, _A, _R = lsnormplanar(self._object_.pts_scn, self._object_.pts_obj, 'stdv')
                #MiWini = np.dot(Mimini, camera_params.inv_wNm)
                #print "Mimini/h_mat:\n", Mimini/self._object_.h_mat
                MiWini = np.dot(self._object_.h_mat, camera_params.inv_wNm)
                # Get the initial pose
                poseparams = getposeparam_sturm(
                    camera_params.camera_matrix, MiWini, "CTW",
                    camera_params.inv_camera_matrix)
                
                # Perform minimization
                if not _minimize_ is None:
                    #print "Performing minimization"
                    #print "Initial pose: ", poseparams
                    XL = np.hstack((camera_params.kvec_scaled,
                                    camera_params.wNm_flat,
                                    camera_params.kscale*self._object_.pts_scn.flatten()))
                    obj_opt_pose = _minimize_(function_cost_total, poseparams,
                    (XL, camera_params.kscale, self._object_.pts_obj),
                    method="L-BFGS-B", tol=1e-3)#"L-BFGS-B") #"Nelder-Mead")
                    if obj_opt_pose.success:
                        poseparams = obj_opt_pose.x
                    #print "Final pose: ", poseparams
                
                retval = 1
                self.sturm_obj_rpy = np.asarray(poseparams[:3][::-1])
                self.sturm_obj_trans = np.asarray(poseparams[3:])
            
            if USE_OPENCV:
                # Solve perspective n-point
                camera_matrix = np.asarray(
                        self.camera.projection_matrix()[:3, :3], order='C')
                retval, rvec, tvec = cv2.solvePnP(self._object_.corners_3d,
                    self.obj_corners, camera_matrix, np.empty(0))
                    #np.asarray(self.camera.distortionCoeffs()))
                # Convert the rotation vector to RPY
                r_mat = cv2.Rodrigues(rvec)[0]
                self.opencv_obj_rpy = np.asarray(
                    tf.transformations.euler_from_matrix(r_mat))
                self.opencv_obj_trans = np.squeeze(tvec)
                if not USE_STURM:
                    self.obj_trans = self.opencv_obj_trans
                    self.obj_rpy = self.opencv_obj_rpy
            
            # Sturm and OpenCV poses computed only if CROSS_VERIFY=True
            if CROSS_VERIFY:
                trans_err = np.abs(self.sturm_obj_trans-self.opencv_obj_trans)
                rpy_err = np.abs(normalize_angle(
                    self.sturm_obj_rpy-self.opencv_obj_rpy))
                if ((np.any(trans_err > VERIFY_MAX_ERR_M)) or
                    (np.any(rpy_err > VERIFY_MAX_ERR_RAD))):
                    print "Cross verify failed"
                    print "Sturm pose :\n", (
                        np.hstack((self.sturm_obj_trans, self.sturm_obj_rpy)))
                    print "OpenCV pose:\n", (
                        np.hstack((self.opencv_obj_trans, self.opencv_obj_rpy)))
                    print "Error:\n", np.hstack((trans_err, rpy_err))
                    self.obj_trans = self.opencv_obj_trans
                    self.obj_rpy = self.opencv_obj_rpy
                else:
                    print "Verification passed."
                    self.obj_trans = self.sturm_obj_trans
                    self.obj_rpy = self.sturm_obj_rpy
            
            if not camera_params.wNm is None:
                COMPUTE_FAKE_COV = False
                # Compute estimate of the covariance
                try:
                    self.obj_cov = self.covariance(True, None)
                except LinAlgError:
                    print "Error computing covariance"
                    exc_info = sys.exc_info()
                    print traceback.print_tb(exc_info[2])
                    COMPUTE_FAKE_COV = True
            else:
                COMPUTE_FAKE_COV = True
            # Fake uncertainty if covariance computation failed/not possible
            if COMPUTE_FAKE_COV:
                # Fake the covariance
                self.obj_cov = 0.0175*np.eye(6)
                pos_diag_idx = range(3)
                self.obj_cov[pos_diag_idx, pos_diag_idx] = (
                    (((1.2*np.linalg.norm(self.obj_trans))**2)*0.03)**2)
            obj_range = np.linalg.norm(self.obj_trans)
            near = self._object_.near_distance
            far = self._object_.far_distance
            if not retval or (not near < obj_range < far):
                print "Too close/far for reliable estimation"
            else:
                detected = True
                # Plot outline on image
                #corners = self.obj_corners.astype(np.int32)
                #cv2.polylines(self.get_scene(0), [corners],
                #              True, (255, 255, 255), 4)
        else:
            self.obj_trans = np.zeros(3)
            self.obj_rpy = np.zeros(3)
            self.obj_cov = np.zeros((6, 6))
        return detected, self.obj_trans, self.obj_rpy, self.obj_cov
    
    def covariance(self, USE_RANDOM_POINTS=False, NUM_POINTS=None):
        camera_params = self._camera_params_
        matches = self._object_
        # Noise variance for each pixel
        px_noise_var = 2.
        if not USE_RANDOM_POINTS:
            # Too expensive to evaluate covariance for all points - 
            # use a subset corresponding to corners and centre of panel
            num_points = 5
            
            pts_obj = np.asarray(matches.pts_obj)
            pts_scn = np.asarray(matches.pts_scn)
            # S� se est� a considerar ruido independente nas coordenadas
            # Select points from pts_obj which are closest to the corners
            # and centre of the template
            height, width = matches.template.shape[:2]
            test_points = np.vstack((matches.corners_2d,
                                     [[width/2., height/2.]]))
            residuals = pts_obj[np.newaxis] - test_points[:, np.newaxis]
            distance = (residuals**2).sum(axis=2)
            test_points_idx = distance.argmin(axis=1)
            # Select the corner points from the template and scene
            cov_pts_obj = pts_obj[test_points_idx]
            cov_pts_scn = pts_scn[test_points_idx]
            
        else:
            # Assume order of the points is random, then select points with
            # linear spacing.
            # Number of matched points
            num_matched_points = matches.pts_obj.shape[0]
            # Select at most 10(?) points to estimate the covariance
            max_num_points = num_matched_points
            if NUM_POINTS is None:
                max_num_points = num_matched_points
            else:
                max_num_points = NUM_POINTS
            num_points = min((max_num_points, num_matched_points))
            
            # The first 13 parameters correspond to camera parameters
            xl_range = np.unique(np.round(np.linspace(0, num_matched_points-1, 
                                                  num_points))).astype(np.int)
            cov_pts_scn = matches.pts_scn[xl_range]
            cov_pts_obj = matches.pts_obj[xl_range]
        
        XL = np.hstack((camera_params.kvec_scaled, camera_params.wNm_flat,
                        camera_params.kscale*cov_pts_scn.flatten()))
        cov_x_coor = px_noise_var*camera_params.kscale**2*np.eye(num_points*2)
        # Adicionar covari�ncia dos par�metros intr�nsecos
        cov_XL = scipy.linalg.block_diag(
            camera_params.camera_cov_scaled, camera_params.wNm_cov, cov_x_coor)
        # Jitter(?) value for calculating the Hessian
        h = 1e-6
        # Calculo da Hessiana de F em ordem a Tetha
        Theta = np.hstack((self.obj_rpy[::-1], self.obj_trans))
        d2FdTheta2 = function_d2FdTheta2(Theta, XL, camera_params.kscale,
                                    cov_pts_obj, h)
        #
        d2FdThetadXL = function_d2FdThetadXL(Theta, XL, camera_params.kscale,
                                        cov_pts_obj, h)
        
        inv_d2FdTheta2 = np.linalg.inv(d2FdTheta2)
        pose_cov_ypr_xyz = np.dot(inv_d2FdTheta2, np.dot(d2FdThetadXL.T, 
            np.dot(cov_XL, np.dot(d2FdThetadXL, inv_d2FdTheta2))))
        
        # Reorder the matrix to x, y, z, roll, pitch, yaw
        cov_xyz = pose_cov_ypr_xyz[3:, 3:]
        cov_rpy = pose_cov_ypr_xyz[:3, :3][::-1, ::-1]
        cov_xyz_rpy = pose_cov_ypr_xyz[3:, :3][:, ::-1]
        cov_pose = np.empty((6, 6))
        cov_pose[:3, :3] = cov_xyz
        cov_pose[3:, 3:] = cov_rpy
        cov_pose[:3, 3:] = cov_xyz_rpy
        cov_pose[3:, :3] = cov_xyz_rpy.T
        return cov_pose
    
    def get_scene(self, idx=None):
        if not idx is None:
            return (self._scene_,)[idx]
        else:
            return (self._scene_,)
    
    def set_detector_num_features(self, num_features):
        self.camera.set_detector_num_features(num_features)
    
    #def show(self):
    #    cv2.namedWindow("panel-detect")
    #    if not self._object_.h_mat is None:
    #        print "Detected the panel!"
    #    #else:
    #    #    print "No detection."
    #    if not self._scene_ is None:
    #        cv2.imshow("panel-detect", self._scene_)
    

class Stereo_detector(Detector):
    def __init__(self, feat_detector=image_feature_extractor.orb):
        # Initialise the detector using the parent class constructor
        super(Stereo_detector, self).__init__(feat_detector=feat_detector)
        self.camera = cameramodels.StereoCameraFeatureDetector(feat_detector)
        # Additional homography for second camera
        self.template_z_offset = 0
        self._object_.keypoints_2d = None
        self._object_.keypoints_3d = None
        self._object_.affine_3d = None
    
    def init_camera(self, camera_info_left, camera_info_right):
        self.camera.fromCameraInfo(camera_info_left, camera_info_right)
        # Additional parameters for the stereo detection
        proj_matrix = self.camera.projection_matrix()
        self.camera_half_baseline = (
            np.abs(proj_matrix[1, 0, 3]/proj_matrix[1, 0, 0])/2)
    
    def set_template(self, template_im=None, corners_3d=None, template_mask=None):
        super(Stereo_detector, self).set_template(template_im, corners_3d, template_mask)
        if len(self._object_.keypoints) == 0:
            rospy.logfatal("No keypoints detected in template image")
            raise rospy.ROSException("No keypoints detected in template image")
        self._object_.keypoints_2d = np.array(
            [_kp_.pt for _kp_ in self._object_.keypoints])
        keypoints_normalised = (self._object_.keypoints_2d/
                                [template_im.shape[1], template_im.shape[0]])
        keypoints_normalised -= 0.5
        # Find size of the template by taking the difference of the corners
        template_size = np.max(np.abs(np.diff(corners_3d, axis=0)), axis=0)
        keypoints_3d = np.hstack((keypoints_normalised,
            np.zeros((keypoints_normalised.shape[0], 1))))
        keypoints_3d *= template_size
        self._object_.keypoints_3d = keypoints_3d
    
    def detect(self, im_scene_l, im_scene_r):
        """
        detect(self, im_scene) -> None
        Detects object by matching features and calculating the homography
        matrix from the template and current scene.
        """
        # Set affine transformation and status to false
        self._object_.status = [False, False]
        self._object_.h_mat = [None, None]
        self._object_.num_inliers = [0, 0]
        self._object_.inliers_status = [None, None]
        self.obj_rpy = np.zeros(3)
        self.obj_trans = np.zeros(3)
        self.obj_corners = np.empty(0)

        # Threshold the image
        self._scene_ = self.process_images((im_scene_l, im_scene_r))

        if self._object_.template is None:
            print "object template is not set!"
            return None

        status_l, h_mat_l, num_inliers_l, inliers_status_l = self._detect_(im_scene_l)
        status_r, h_mat_r, num_inliers_r, inliers_status_r = self._detect_(im_scene_r)
        self._object_.status = [status_l, status_r]
        self._object_.h_mat = [h_mat_l, h_mat_r]
        self._object_.num_inliers = [num_inliers_l, num_inliers_r]
        self._object_.inliers_status = [inliers_status_l, inliers_status_r]
    
    def location(self):
        """
        Return detected (bool), relative position (x, y, z) and
        orientation (RPY) of the object.
        """
        status = self._object_.status
        h_mat = self._object_.h_mat
        num_inliers = np.asarray(self._object_.num_inliers)
        #inliers_status = self._object_.inliers_status
        obj_corners = []
        obj_trans = []
        obj_rpy = []
        for idx in range(2):
            if status[idx]:
                _obj_corners_ = cv2.perspectiveTransform(
                    self._object_.corners_2d.reshape(1, -1, 2),
                    h_mat[idx]).reshape(-1, 2)
                # Solve perspective n-point
                _retval_, _rvec_, _tvec_ = cv2.solvePnP(
                    self._object_.corners_3d, _obj_corners_,
                    self.camera.camera_matrix(), np.empty(0))
                # Convert the rotation vector to RPY
                _r_mat_ = cv2.Rodrigues(_rvec_)[0]
                _rpy_ = np.asarray(tf.transformations.euler_from_matrix(_r_mat_))
                # Check if object in valid range
                obj_range = np.linalg.norm(_tvec_)
                if not _retval_ or (not 0.25 < obj_range < 5):
                    status[idx] = False
                else:
                    # Plot the bounding box
                    corners = _obj_corners_.astype(np.int32)
                    cv2.polylines(self.get_scene(idx), [corners],
                                  True, (255, 255, 255), 4)
            else:
                _obj_corners_ = np.zeros((1, 3))
                _rpy_ = np.zeros(3)
                _tvec_ = np.zeros(3)
            obj_corners.append(_obj_corners_)
            obj_trans.append(np.squeeze(_tvec_))
            obj_rpy.append(_rpy_)

        # Convert the translation from the right image to the left
        tf_obj_trans = self.camera.left.from_world_coords(
            self.camera.right.to_world_coords(obj_trans[1][np.newaxis]))[0]
        detected = status[0] or status[1]
        # Compute the weighted average of the pose
        total_inliers = (num_inliers*status).sum()
        try:
            position = np.squeeze((obj_trans[0]*status[0]*num_inliers[0] +
                    tf_obj_trans*status[1]*num_inliers[1])/total_inliers)
        except:
            print "tf failed!"
            return 0, np.zeros(3), np.zeros(3)
        orientation = (obj_rpy[0]*status[0]*num_inliers[0] +
                       obj_rpy[1]*status[1]*num_inliers[1])/total_inliers
        return detected, position, orientation
    
    def get_scene(self, idx=None):
        if not idx is None:
            return self._scene_[idx]
        else:
            return self._scene_
    

def getposeparam_sturm(K, MiW, wtcflag="CTW", inv_K=None):
    """POSEPARAM = getposeparam(K,MiW)
    Pose Parameters from Homography using Sturm method:
    
    (falta help) 
    Paper Algorithms for Plane--Based Pose Estimation
    bib_nreg.bib - Sturm00
    
    (help antigo)
     
    Calcula os par�metros de pose de uma camera com matrix de parametros
    intr�nsecos K, e com uma homografia MiW entre a imagem e o plano de
    refer�ncia do mundo. Devolve os par�metros de pose
    POSEPAR = [a b c tx ty tz]. Os �ngulos a, b e c s�o os �ngulos 
    fixos X-Y-Z tal como definidos no Craig 2� Ed. Pag 46, para a rota��o 
    cRw. Os elementos tx, ty e tz correspondem � transla��o do ref da camera
    cTw. Para cada valor de MiW e K existem duas solu��es possiveis para a
    pose. Esta fun��o devolve a solu��o correspondente � componente positiva
    de wTc segundo ZZ, i.e., centro �ptico acima do plano do referencial do
    mundo. O referencial 3D do plano � coincidente com o 2D com o eixo
    adicional ZZ definido da maneira convencional a partir dos dois outros. 
    
    POSEPARAM = getposeparam(K,MiW,'WTC')
    Devolve os par�metros de pose na forma
    POSEPAR = [a b c wtcx wtcy wtcz], na qual os elementos wtcx, 
    wtcy e wtcz correspondem � transla��o do ref da camera wTc.
    
    Efectua a opera��o inversa de POSE1HOMO.M
    """ 

    if (K[1, 0]**2 + K[2, 0]**2 + K[2, 1]**2) > 1e9:
        print 'K should be a upper triangular matrix on GETPOSEPARAM.'
    wtcflag = wtcflag.upper()
    assert wtcflag in ["CTW", "WTC"], "wtcflag must be 'CTW' or 'WTC'"
    
    if inv_K is None:
        inv_K = np.linalg.inv(K)
    A = np.dot(inv_K, MiW)
    
    # Compute the "economy size" SVD
    U, S, V = np.linalg.svd(A[:, :2], full_matrices=0)
    
    L = np.dot(U, V.T)
    
    ln3 = np.cross(L[:, 0], L[:, 1])
    CRW1 = np.hstack((L, ln3[:, np.newaxis]))
    
    #Lambda = trace(S)/2;
    Lambda = S.sum()/2.
    
    CTWORG1 = A[:, 2] / Lambda
    WTCORG1 = np.dot(-CRW1.T, CTWORG1)
    
    #if WTCORG1[2] > 0:
    #    CRW2 = np.dot(CRW1, np.diag([-1., -1., 1.]))
    #    WTCORG2 = np.dot(np.diag([1., 1., -1.]), WTCORG1)
    #else:
    #    CRW2 = CRW1
    #    WTCORG2 = WTCORG1
    #    CRW1 = np.dot(CRW1, np.diag([-1., -1., 1.]))
    #    WTCORG1 = np.dot(np.diag([1., 1., -1.]), WTCORG1)
    #    CTWORG1 = -CTWORG1
    
    a, b, c = rot2xyzfixed(CRW1)
    
    if wtcflag == 'CTW':
        POSEPARAM = np.hstack((a, b, c, CTWORG1[:3]))
    else:
        POSEPARAM = np.hstack((a, b, c, WTCORG1[:3]))
    
    return POSEPARAM


def rot2xyzfixed(ROT3):
    """a, b, c = rot2xyzfixed(ROT3)
    Rotation to X-Y-Z fixed angles
    Calcula os �ngulos fixos de rota��o X-Y-Z a partir de uma matriz de 
    rota��o 3x3, de acordo com as f�rmulas do Craig 2� Ed. Pag. 47.
    """
    assert ROT3.shape == (3, 3), "Rotation matrix must be a 3x3 matrix"
    
    b = np.arctan2(-ROT3[2, 0], np.sqrt((ROT3[0, 0])**2 + (ROT3[1, 0])**2))
    cos_b = np.cos(b)
    # Check for degeneracy
    if abs(cos_b) < 1e-9:
        print "Degenerate angle set on angle extraction from rotation matrix."
        print "Cos(b) = ", cos_b
        b = np.sign(b) * np.pi/2
        a = 0.
        c = np.sign(b) * np.arctan2(ROT3[0, 1], ROT3[1, 1])
    else:
        a = np.arctan2(ROT3[1, 0]/cos_b, ROT3[0, 0]/cos_b)
        c = np.arctan2(ROT3[2, 1]/cos_b, ROT3[2, 2]/cos_b)
    return a, b, c


def function_cost_total(Thetain, XLin, kscale, COORm):
    """funcao de custo usando os parametros intrinsecos e a colineacao que 
    calibra metricamente o mosaico"""
    K = np.asarray([[XLin[0], 0, XLin[2]],
                    [0, XLin[1], XLin[3]],
                    [0, 0, kscale]])
    wNm = XLin[4:13].reshape((3, 3))
    TiW = pose1homo(Thetain, K, 'CTW')
    
    image_points = XLin[13:]/kscale
    num_points = image_points.shape[0]/2
    image_points = image_points.reshape((num_points, 2))
    assert 1 <= Thetain.ndim <= 2, ("Thetain must be 1D for single pose or \
        2D for mutiple poses")
    if Thetain.ndim == 1:
        PSI = np.dot(TiW, wNm)
        PSI /= np.linalg.norm(PSI, 'fro')
        
        COORm_em_i = trancoor(PSI, COORm)
        fcost = np.linalg.norm(image_points - COORm_em_i, 'fro')#/1000.
    else:
        PSI = np.asarray([np.dot(_TiW_, wNm) for _TiW_ in TiW])
        PSI_norm = np.asarray(map(np.linalg.norm, PSI))
        PSI /= PSI_norm[:, np.newaxis, np.newaxis]
        
        COORm_em_i = trancoor(PSI, COORm)
        fcost = np.asarray(map(np.linalg.norm, image_points - COORm_em_i))#/1000.
    
    return fcost


def pose1homo(poseparams, K, wtcflag="CTW"):
    """TiW = pose1homo(POSEPAR,K)
    Homography from Pose Parameters
    
    Cria uma homografia TiW a partir de par�metros de pose.
    Usa os seguintes par�metros de pose POSEPAR = [a b c tx ty tz].
    Os �ngulos a, b e c s�o os �ngulos fixos X-Y-Z tal como definidos no
    Craig 2� Ed. Pag 46, para a rota��o cRw. Os elementos tx, ty e tz
    correspondem � transla��o do ref da camera cTw.
    
    TiW = pose1homo(POSEPAR,K,'WTC')
    Usa os par�metros de pose na forma POSEPAR = [a b c wtcx wtcy wtcz], na qual os elementos wtcx, 
    wtcy e wtcz correspondem � transla��o do ref da camera wTc.
    """
    
    wtcflag = wtcflag.upper()
    assert wtcflag in ["WTC", "CTW"], "wtcflag must be 'CTW' or 'WTC'"
    if poseparams.ndim == 1:
        a, b, c = poseparams[:3]
        cos_a, cos_b, cos_c = np.cos((a, b, c))
        sin_a, sin_b, sin_c = np.sin((a, b, c))
        
        phi_1 = np.asarray((cos_a*cos_b, sin_a*cos_b, -sin_b))
        phi_2 = np.asarray((cos_a*sin_b*sin_c-sin_a*cos_c, 
                            sin_a*sin_b*sin_c+cos_a*cos_c, cos_b*sin_c))
        
        if wtcflag == 'CTW':
            ctw = poseparams[3:]
        else:
            wtc = poseparams[3:]
            phi_3 = np.asarray((cos_a*sin_b*cos_c+sin_a*sin_c,
                                sin_a*sin_b*cos_c-cos_a*sin_c, cos_b*cos_c))
            ctw = np.dot(-np.vstack((phi_1, phi_2, phi_3)).T, wtc)
        
        TiW = np.dot(K, np.vstack((phi_1, phi_2, ctw)).T)
        TiW /= TiW[2, 2]
    else:
        num_poses = poseparams.shape[0]
        TiW = np.empty((num_poses, 3, 3), dtype=np.double)
        phi_1 = np.empty((num_poses, 3), dtype=np.double)
        phi_2 = np.empty((num_poses, 3), dtype=np.double)
        
        cos_pose = np.cos(poseparams[:, :3])
        sin_pose = np.sin(poseparams[:, :3])
        cos_a = cos_pose[:, 0]; cos_b = cos_pose[:, 1]; cos_c = cos_pose[:, 2]
        sin_a = sin_pose[:, 0]; sin_b = sin_pose[:, 1]; sin_c = sin_pose[:, 2]
        
        phi_1[:, 0] = cos_a*cos_b
        phi_1[:, 1] = sin_a*cos_b
        phi_1[:, 2] = -sin_b
        
        phi_2[:, 0] = cos_a*sin_b*sin_c-sin_a*cos_c
        phi_2[:, 1] = sin_a*sin_b*sin_c+cos_a*cos_c
        phi_2[:, 2] = cos_b*sin_c
        
        phi_aug = np.empty((num_poses, 3, 3), dtype=np.double)
        phi_aug[:, :, 0] = phi_1
        phi_aug[:, :, 1] = phi_2
        if wtcflag == 'CTW':
            ctw = poseparams[:, 3:]
        else:
            wtc = poseparams[:, 3:]
            phi_3 = np.empty((num_poses, 3), dtype=np.double)
            phi_3[:, 0] = cos_a*sin_b*cos_c+sin_a*sin_c
            phi_3[:, 1] = sin_a*sin_b*cos_c-cos_a*sin_c
            phi_3[:, 2] = cos_b*cos_c
            phi_aug[:, :, 2] = phi_3
            ctw = np.asarray([np.dot(-_phi_, _wtc_)
                for (_phi_, _wtc_) in zip(phi_aug, wtc)])
        phi_aug[:, :, 2] = ctw
        TiW = np.asarray([np.dot(K, _phi_) for _phi_ in phi_aug])
        TiW /= TiW[:, 2, 2, np.newaxis, np.newaxis]
    
    return TiW

def function_d2FdTheta2(Theta, XL, kscale, COORm, h):
    """
    d2FdTheta2 = function_d2FdTheta2(Theta, XL, kscale, COORm, h)
    Calculo da Hessiana de F em ordem a Tetha
    """
    num_theta = Theta.shape[0]
    half_h = h/2.
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    #log_d2FdTheta2 = np.empty((num_theta, num_theta))
    #sign_d2FdTheta2 = np.empty((num_theta, num_theta))
    #poses = np.empty((4, num_theta), dtype=np.double)
    derivative_sign = np.asarray((+1, -1, -1, +1), dtype=np.double)
    
    #Theta_vec = np.repeat([Theta], 4, axis=0)
    half_h_vec1 = np.asarray([half_h, half_h, -half_h, -half_h])
    half_h_vec2 = np.asarray([half_h, -half_h, half_h, -half_h])
    
    # Generate all Theta and reshape as N x (Px4) x NumTheta
    Theta_vec_all = np.repeat([Theta], num_theta*num_theta*4, axis=0).reshape((num_theta, num_theta*4, num_theta))
    half_h_vec1_all_p = np.repeat([half_h_vec1], num_theta, axis=0).flatten()
    half_h_vec2_all_p = (np.repeat(np.eye(6), 4, 0).reshape((6, 4, 6))*half_h_vec2[np.newaxis, :, np.newaxis]).reshape((num_theta*4, num_theta))
    derivative_sign_all_p = derivative_sign*np.ones((6, 1))
    log_d2FdTheta2_all_p = log_inv_hsq + np.zeros((num_theta, num_theta))
    sign_d2FdTheta2_all_p = np.empty((num_theta, num_theta))
    for n in range(num_theta):
        Theta_vec_all[n, :, n] += half_h_vec1_all_p
        Theta_vec_all[n] += half_h_vec2_all_p
        func_cost = function_cost_total(Theta_vec_all[n], XL, kscale, COORm).reshape((num_theta, 4))
        P_sum = (func_cost*derivative_sign_all_p).sum(axis=1)
        P_sign = sign(P_sum)
        log_d2FdTheta2_all_p[n] += log(np.abs(P_sum))
        sign_d2FdTheta2_all_p[n] = P_sign
    d2FdTheta2 = np.exp(log_d2FdTheta2_all_p)*sign_d2FdTheta2_all_p
    return d2FdTheta2


def function_d2FdThetadXL(Theta, XL, kscale, mosaic_points, h):
    # Calculo da segunda derivada de F em ordem a Tetha e PSI
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    half_h = h/2.
    log_d2FdThetadXL = log_inv_hsq*np.ones((XL.shape[0], Theta.shape[0]))
    sign_d2FdThetadXL = np.zeros((XL.shape[0], Theta.shape[0]))
    #print "compute_pose_unc():100"
    #code.interact(local=locals())
    
    num_theta = Theta.shape[0]
    half_h_delta = np.eye(num_theta)*half_h
    Theta_plus = np.repeat([Theta], num_theta, axis=0)
    Theta_minus = Theta_plus - half_h_delta
    Theta_plus += half_h_delta
    XL_plus = XL.copy()
    XL_minus = XL.copy()
    max_num_points = XL.shape[0]-13
    theta_range = np.arange(13)
    xl_range = np.unique(np.round(np.linspace(13, XL.shape[0]-1, max_num_points))).astype(np.int) #range(XL.shape[0])
    n_range = np.hstack((theta_range, xl_range))
    for n in n_range: #1:length(XL)
        old_XL = XL[n]
        XL_plus[n] += half_h
        XL_minus[n] -= half_h
        for p in range(Theta.shape[0]): #1:length(Theta)
            #pose_vecs = np.asarray((vadd(Theta, p,  half_h), vadd(Theta, p, -half_h), vadd(Theta, p,  half_h), vadd(Theta, p, -half_h)))
            #xl_vecs = np.asarray((vadd(XL, n,  half_h), vadd(XL, n,  half_h), vadd(XL, n, -half_h), vadd(XL, n, -half_h)))
            P1 = function_cost_total(Theta_plus[p], XL_plus, kscale, mosaic_points)
            P2 = function_cost_total(Theta_minus[p], XL_plus, kscale, mosaic_points)
            P3 = function_cost_total(Theta_plus[p], XL_minus, kscale, mosaic_points)
            P4 = function_cost_total(Theta_minus[p], XL_minus, kscale, mosaic_points)
            #% troca de n e p em relacao a numxdiff.m
            p1_p4 = P1 + P4
            p2_p3 = P2 + P3
            P_sum = p1_p4 - p2_p3
            P_sign = sign(P_sum)
            log_d2FdThetadXL[n, p] += log(abs(P_sum))
            sign_d2FdThetadXL[n, p] = sign(P_sign)
        XL_plus[n] = old_XL
        XL_minus[n] = old_XL
    d2FdThetadXL = sign_d2FdThetadXL*np.exp(log_d2FdThetadXL)
    return d2FdThetadXL


def trancoor(transformation, coords):
    """TCOOR = trancoor(T, COOR)
    Apply Coordinate transformation:
    Aplica a transforma��o T(3x3) � lista COOR(nx2).
    """
    num_points = coords.shape[0]
    # Create homogeneous list
    coords = np.hstack((coords, np.ones((num_points, 1))))
    assert 2 <= transformation.ndim <= 3, (
        "transformation matrix must be 2D for single transform, \
        or 3D for multiple transforms")
    if transformation.ndim == 2:
        # Get transformed list
        tr_coords = np.dot(coords, transformation.T)
        
        # Divide by the homogeneous parameter
        tr_coords = tr_coords[:, :2]/tr_coords[:, np.newaxis, 2]
    else:
        tr_coords = np.asarray([np.dot(coords, _transformation_.T)
            for _transformation_ in transformation])
        tr_coords = tr_coords[:, :, :2]/tr_coords[:, :, 2, np.newaxis]
    
    return tr_coords


def lsnormplanar(COOR1, COOR2, methodnorm):
    """M12 = lsnormplanar(COOR1, COOR2, methodnorm)
    Least Squares Normalized Planar Transformation
    
    Determina��o da Transforma��o Planar M12(3x3) que relaciona os pontos 
    com coordenadas COOR1 e COOR2. Esta fun��o normaliza as coordenadas 
    dos pontos impondo m�dia nula e permite ainda efectuar 
    escalamentos do tipo :
    methodnorm = 'nnor' : (no normalization). 
    methodnorm = 'none' : nenhum escalamento. 
    methodnorm = 'stdv' : desvio padr�o unit�rio em cada coordenada.
    methodnorm = 'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    S�o necess�rias pelo menos 4 pontos em cada lista de coordenadas.
    
    M12, CONDS = lsnormplanar(COOR1, COOR2) devolve tamb�m o n� de 
    condi��o do sistema (A'*A) referente 'as coordenadas normalizadas.
    M12, CONDS, A, R = lsnormplanar(COOR1, COOR2) devolve a matriz do 
    sistema A e o vector de res�duos R referente 'as coordenadas 
    normalizadas.
    
    Usa o sistema (u,v) para as coordenadas 2-D.
    
    M12 = lsplanar(COOR1, COOR2, methodnorm, methodplan)
    Permite a especifica��o de uma outra transforma��o planar:
      methodplan = 
          'lsplanar' : Transforma��o planar gen�rica.
    """
    
    assert COOR1.shape == COOR2.shape, (
        "Coordinate lists must have the same size.")
    
    # Normalize the coordinates
    NCL1, T1 = normcoor(COOR1, methodnorm)
    NCL2, T2 = normcoor(COOR2, methodnorm)
    
    # Call the usual planar transformation estimation function
    M12norm, CONDS, A, R = lsplanar(NCL1, NCL2, 'none')
    
    # Invert the effect of the normalization
    M12 = np.dot(np.linalg.inv(T1), np.dot(M12norm, T2))
    
    M12 /= M12[2, 2]
    #print "lsnormplanar():296"
    #code.interact(local=locals())
    
    return M12, CONDS, A, R


def normcoor(COOR, METHOD="none"):
    """NCOOR, T = normcoor(COOR)
    Normalize 2D Coordinates:
    
    Normaliza uma lista COOR de coordenadas 2D, tal que tenha m�dia nula.
    Devolve lista modificada NCOOR e a tranforma��o homog�nea afim T(3x3) 
    correspondente.
    
    NCOOR, T = normcoor(COOR, METHOD)
    Permite efectuar escalamentos do tipo :
      METHOD =
      'nnor' : (no normalization). Devolve as coordenadas de entrada,
               com T = eye(3)
      'none' : nenhum escalamento. 
      'stdv' : desvio padr�o unit�rio e independente em cada coordenada.
      'equa' : desvio padr�o aproximadamente unit�rio e igual para as duas
               coordenadas (m�dia dos desvios padroes).
      'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    
    COOR = trancoor(inv(T),NCOOR)
    """
    METHOD = METHOD.lower()
    assert METHOD in ("nnor", "stdv", "equa", "sqrt", "none"), (
        'METHOD must be one of "nnor", "stdv", "equa", "sqrt", "none"')
    if METHOD == "nnor":
        # Devolver as coordenadas de entrada 
        NCOOR = COOR
        T = np.eye(3)
        return NCOOR, T
    
    assert COOR.shape[1] == 2, "Coordinate array error"
    
    # Create the list of the repeated vector mean 
    MCOOR = np.mean(COOR, axis=0)
    
    if METHOD == "stdv":
        # Compute the vector standard deviation
        SDCOOR = np.mean((COOR - MCOOR)**2, axis=0)**0.5
    elif METHOD == 'equa':
        # Compute the average standard deviation
        SDCOOR = np.mean(
            np.mean((COOR - MCOOR)**2, axis=0), axis=0)*np.ones(2)
    elif METHOD == "sqrt":
        # Compute the sqr of the vector standard deviation
        SDCOOR = np.mean((COOR - MCOOR)**2, axis=0)
    elif METHOD == "none":
        SDCOOR = np.ones(2)
    
    # Account for data degeneracies, i. e., null st. dev.
    SDCOOR += np.any(SDCOOR < 1e-3) * 1e-3
    
    # Compute the homogeneous transformation
    transformation = np.eye(3)
    transformation[0, 0] = SDCOOR[0]**(-1)
    transformation[1, 1] = SDCOOR[1]**(-1)
    transformation[:2, 2] = -(MCOOR/SDCOOR)
    
    # Create homogeneous list
    COOR = np.hstack((COOR, np.ones((COOR.shape[0], 1))))
    
    # Get transformed list
    NCOOR = np.dot(COOR, transformation.T)
    
    # Divide by the homogeneous parameter
    NCOOR = NCOOR[:, :2]/NCOOR[:, np.newaxis, 2]
    
    return NCOOR, transformation


def lsplanar(COOR1, COOR2, normmethod='stdv'):
    """M12 = lsplanar(COOR1, COOR2)
    Determina��o da Transforma��o Planar M12(3x3) que relaciona os pontos
    com coordenadas COOR1 e COOR2.
    S�o necess�rias pelo menos 4 pontos em cada lista de coordenadas.
    
    M12, CONDS = lsplanar(COOR1, COOR2) devolve tamb�m o n� de condi��o 
    do sistema (A'*A).
    [M12,CONDS,A,R] = lsplanar(COOR1, COOR2) devolve a matriz do sistema 
    A e o vector de res�duos R.
    
    Usa o sistema (u,v) para as coordenadas 2-D.
    
    M12 = lsplanar(COOR1, COOR2, method)
    Permite a especifica��o de uma outra transforma��o planar:
      method = 'lsplanar' :  Transforma��o planar gen�rica.
    
    M12 = lsplanar(COOR1, COOR2, method, normmethod)
    Permite a especifica��o de um metodo para a normaliza�ao de coordenadas
    antes da estima�ao da transformacao planar:
      normmethod = 
          'none' : nenhum escalamento, so' transla�ao. 
          'stdv' : desvio padr�o unit�rio em cada coordenada.
          'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    """
    
    assert COOR1.shape == COOR2.shape, (
        "Coordinate lists must have the same size.")
    
    num_points = COOR1.shape[0]
    assert num_points > 1, (
        "Coordinate lists must have at least 1 matched points")
    
    # Teste com normaliza��o de coordenadas (parte 1/2)
    NCOOR1, T1 = normcoor(COOR1, normmethod)
    NCOOR2, T2 = normcoor(COOR2, normmethod)
    COOR1 = NCOOR1
    COOR2 = NCOOR2
    
    # No coordinate swapping
    ones_vec = np.ones((num_points, 1))
    COOR1 = np.hstack((COOR1, ones_vec))
    COOR2 = np.hstack((COOR2, ones_vec))
    
    #print "lsplanar():413"
    #code.interact(local=locals())
    
    A = np.zeros((2*num_points, 9))
    
    mat_1 = np.asarray([[1., 0., 0.]])
    mat_2 = np.asarray([[0., 1., 0.]])
    for l in range(num_points):
        mat_1[0, 2] = -COOR1[l, 0]
        mat_2[0, 2] = -COOR1[l, 1]
        ML1 = np.dot(COOR2[l, np.newaxis].T, mat_1)
        ML2 = np.dot(COOR2[l, np.newaxis].T, mat_2)
        A[2*l] = ML1.T.flatten()
        A[2*l+1] = ML2.T.flatten()
    
    #print "lsplanar():428"
    #code.interact(local=locals())
    
    # get the unit eigvector of A'*A corresponding to the smallest
    # eigenvalue of A'*A.
    N, CONDS = uniteig(A)
    if np.isinf(CONDS):
        print('A condi��o de A�*A � infinita !')
    
    # compute the residuals
    R = np.dot(A, N)
    #M12 = reshape(N(:,1),3,3)';
    M12 = N.reshape((3, 3))
    
    # Teste com normaliza��o de coordenadas (parte 2/2)
    M12 = np.dot(np.linalg.inv(T1), np.dot(M12, T2))
    
    #print "lsplanar():441"
    #code.interact(local=locals())
    return M12, CONDS, A, R


def uniteig(A):
    """N = uniteig(A)
    Get Unit Eigenvector of A'*A :
    Devolve o vector pr�prio unit�rio de A'*A correspondente ao menor
    valor pr�prio de A�*A. 
    
    N,CONDNUM = uniteig(A) :  Devolve tamb�m o n� de condi��o de A'*A. 
    """
    # get the unit eigvector of A'*A corresponding to the 
    # smallest eigenvalue of A.
    AT_A = np.dot(A.T, A)
    D, V = np.linalg.eigh(AT_A)
    I = D.argmin()
    N = V[:, I]
    N /= np.sqrt((N**2).sum())
    
    CONDNUM = np.linalg.cond(AT_A)
    
    if np.isinf(CONDNUM):
        print('UNITEIG : A condi��o de A�*A � infinita !')
    #print "uniteig():461"
    #code.interact(local=locals())
    return N, CONDNUM


def mvee(points, tol = 0.001):
    """
    http://stackoverflow.com/a/14025140
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)        
    A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    return A, c

def ellipse_extreme_points(A, c):
    """
    extreme_points = ellipse_extreme_points(A, c)
    Using the Ellipse to Fit and Enclose Data Points
    A First Look at Scientific Computing and Numerical Optimization
    Charles F. Van Loan
    Department of Computer Science, Cornell University
    http://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf
    """
    pass



def wnm_cost(wnm, camera_matrix, mimini, target_pose):
    wnm_mat = np.reshape(wnm, (3, 3))
    inv_wnm = np.linalg.inv(wnm_mat)
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    miwini = np.dot(mimini, inv_wnm)
    poseparams = getposeparam_sturm(camera_matrix, miwini, "WTC", inv_camera_matrix)
    poseparams_xyz = poseparams[3:]
    poseparams_rpy = poseparams[:3][::-1]
    cost = np.linalg.norm(poseparams_xyz - target_pose[:3])
    cost += np.linalg.norm(normalize_angle(poseparams_rpy - target_pose[3:]))
    return cost

##############################################################################
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
