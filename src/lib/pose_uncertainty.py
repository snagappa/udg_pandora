#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numpy.linalg import norm
from scipy.linalg import block_diag
import scipy.optimize
import scipy.io

import cv2
import roslib
roslib.load_manifest("tf")
from tf.transformations import euler_from_matrix
from lib.misctools import STRUCT
from lib import cameramodels, image_feature_extractor

import code

_minimize_ = getattr(scipy.optimize, "minimize", None)

class PoseDetector(object):
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
        self._object_.keypoints_3d = None
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
        # Store matches
        self._object_.matches = STRUCT()
        self._object_.matches.obj_keypoints_2d = None
        self._object_.matches.obj_keypoints_3d = None
        self._object_.matches.scn_keypoints = None
        self._object_.matches.XL = None
        #
        self._object_.wNm = np.eye(3)
        self._object_.inv_wNm = np.eye(3)
        
        # Camera
        self.camera = cameramodels.PinholeCameraFeatureDetector(feat_detector)
        self.kscale = 0
        self.camera_matrix = None
        self.kvec_scaled = None
        self.camera_cov_scaled = None
        self.inv_camera_matrix = None
        self.pixel_noise = 1.
        self._var_pixel_ = None
        
        # Optimisation result for pose estimation
        self.obj_opt_pose = None
        # Rotation matrix and translation vector from solvePnP
        self.obj_rpy = None
        self.obj_rpy_cov = None
        self.obj_trans = None
        self.obj_corners = np.empty(0)
        # Image of the current scene
        self._scene_ = None
        # Set up FLANN matcher
        self.flann_ratio_threshold = 0.75
    
    def init_camera(self, camera_info, *dummy_args):
        self.camera.fromCameraInfo(camera_info)
        camera_matrix = self.camera.projection_matrix[:, :3]
        # Normalise(?) camera matrix
        camera_matrix /= camera_matrix[2, 2]
        # Normalisation of camera matrix covariance (use Frobenius norm)
        kscale = 1./np.linalg.norm(camera_matrix) 
        self.camera_matrix = camera_matrix
        self.kscale = kscale
        self.kvec_scaled = camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]] * kscale
        self.camera_cov_scaled = camera_cov * kscale**2
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
    
    def get_wNm(self):
        return self._object_.wNm.copy()
    
    def set_wNm(self, wNm, wNm_cov=None):
        assert np.abs(np.linalg.norm(wNm) - 1) < 1e-6, "Norm of wNm must be 1"
        self._object_.wNm = wNm.copy()
        self.wNm_flat = wNm.flatten()
        self._object_.inv_wNm = np.linalg.inv(wNm)
        if wNm_cov is None:
            self.wNm_cov = np.zeros((9, 9))
        else:
            self.wNm_cov = wNm_cov.copy()
    
    def set_pixel_noise(self, pixel_noise, matrix_size=200):
        self.pixel_noise = pixel_noise
        self._var_pixel_ = np.eye(matrix_size)*pixel_noise
    
    def set_flann_ratio_threshold(self, flann_ratio_threshold):
        self.flann_ratio_threshold = flann_ratio_threshold
    
    def set_template(self, template_im=None, corners_3d=None,
                     template_mask=None):
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
        self.calc_keypoints_3d()
    
    def set_corners3d(self, corners_3d):
        if not corners_3d is None:
            assert_err_msg = "corners must a Nx3 ndarray, N>=4"
            assert type(corners_3d) is np.ndarray, assert_err_msg
            assert (corners_3d.ndim==2) and \
                   (corners_3d.shape[1]==3) and \
                   (corners_3d.shape[0]>3), assert_err_msg
            self._object_.corners_3d = corners_3d
            self.calc_keypoints_3d()
    
    def calc_keypoints_3d(self):
        _object_ = self._object_
        if ((not _object_.corners_3d is None) and
            (not _object_.corners_2d is None) and
            (_object_.corners_2d.ndim==2) and 
            (np.prod(_object_.corners_2d.shape))):
            # Define 3D coordinates of the keypoints
            width_height = _object_.corners_2d[2]
            kp_array = np.asarray([__kp__.pt for __kp__ in _object_.keypoints])
            keypoints_normalised = kp_array/width_height
            keypoints_normalised -= 0.5
            # Find size of the template by taking the difference of the corners
            template_size = np.max(
                np.abs(np.diff(_object_.corners_3d, axis=0)), axis=0)
            keypoints_3d = np.hstack((keypoints_normalised,
                np.zeros((keypoints_normalised.shape[0], 1))))
            keypoints_3d *= template_size
            _object_.keypoints_3d = keypoints_3d
    
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
            self._object_.descriptors = np.vstack((self._object_.descriptors,
                                                   extra_descriptors))
            self.calc_keypoints_3d()
    
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
        dam_result = self.camera.detect_and_match(self._object_.keypoints,
                                                  self._object_.descriptors,
                                                  keypoints_scene,
                                                  descriptors_scene,
                                                  self.flann_ratio_threshold)
        pts_obj, pts_scn, obj_valid_idx, scn_valid_idx = dam_result[0:4]
        status, h_mat, num_inliers, inliers_status = (
        self.camera.find_homography(pts_scn, pts_obj, ransacReprojThreshold=10,
            min_inliers=self._object_.min_correspondences))
        matches = self._object_.matches
        try:
            h_mat = np.linalg.inv(h_mat)
            print "Homography matrix:\n", h_mat
        except:
            print "Error computing inverse of homography!"
            h_mat = None
            status = False
            matches.obj_keypoints_2d = None
            matches.obj_keypoints_3d = None
            matches.scn_keypoints = None
        else:
            inliers_idx = scn_valid_idx.astype(np.bool)
            matches.obj_keypoints_2d = pts_obj[inliers_idx]
            matches.obj_keypoints_3d = self._object_.keypoints_3d[obj_valid_idx]
            matches.scn_keypoints = pts_scn[inliers_idx]
            
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
    
    def location(self):
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
        if self._object_.status:
            # Perform minimisation using the initial estimated homography
            MiWini = np.dot(self._object_.h_mat, self._object_.inv_wNm)
    
            # Generate initial pose parameters
            poseparams_ini = getposeparam_sturm(self.camera_matrix, MiWini, 
                                                "WTC", self.inv_camera_matrix)
            
            XL = np.hstack((self.kvec_scaled, self.wNm_flat,
                            self.kscale*self.matches.obj_keypoints_2d.flatten()))
            self._object_.XL = XL
            
            if not _minimize_ is None:
                obj_opt_pose = _minimize_(function_cost_total, poseparams_ini,
                (XL, self.kscale, self.matches.obj_keypoints_2d),
                method="L-BFGS-B", tol=1e-3)#"L-BFGS-B") #"Nelder-Mead")
            else:
                # Refine the pose using Simplex Downhill
                Tfin, fopt, num_iters, num_funcalls, warnflag = (
                    scipy.optimize.fmin(function_cost_total, poseparams_ini,
                    (XL, self.kscale, self.matches.obj_keypoints_2d),
                    xtol=1e-6, ftol=1e-6, full_output=1))
                obj_opt_pose = scipy.optimize.optimize.Result({"x":Tfin,
                    "fun":fopt, "nfev":num_funcalls, "nit":num_iters,
                    "success":not warnflag, "status":warnflag})
            
            self.obj_opt_pose = obj_opt_pose
            if obj_opt_pose.success:
                retval = True
                self.obj_trans = obj_opt_pose.x[3:]
                self.obj_rpy = obj_opt_pose.x[:3][::-1]
            else:
                print "Minimization did not converge!"
                    
                # Project the corners of the template onto the scene
                self.obj_corners = cv2.perspectiveTransform(
                    self._object_.corners_2d.reshape(1, -1, 2),
                    self._object_.h_mat).reshape(-1, 2)
                # Solve perspective n-point
                camera_matrix = np.asarray(
                        self.camera.projection_matrix()[:3, :3], order='C')
                retval, rvec, tvec = cv2.solvePnP(self._object_.corners_3d,
                    self.obj_corners, camera_matrix, np.empty(0))
                    #np.asarray(self.camera.distortionCoeffs()))
                # Convert the rotation vector to RPY
                r_mat = cv2.Rodrigues(rvec)[0]
                self.obj_rpy = np.asarray(euler_from_matrix(r_mat))
                self.obj_trans = np.squeeze(tvec)
            
            obj_range = np.linalg.norm(self.obj_trans)
            if not retval or (not 0.25 < obj_range < 5):
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
        return detected, self.obj_trans, self.obj_rpy, obj_opt_pose.success
    
    def covariance(self, h=1e-6):
        num_points = self.matches.obj_keypoints_2d.shape[0]
        if num_points > self._var_pixel_.shape[0]:
                self.set_pixel_noise(self.pixel_noise, 
                    matrix_size=self.matches.obj_keypoints_2d.shape[0])
        cov_x_coor = self._var_pixel_[:num_points, :num_points]
        # Adicionar covari�ncia dos par�metros intr�nsecos
        cov_XL = block_diag(self.camera_cov_scaled, self.wNm_cov, cov_x_coor)
        XL = self._object_.XL
        Theta = self.obj_opt_pose
        # Calculo da Hessiana de F em ordem a Tetha
        d2FdTheta2 = function_d2FdTheta2(Theta, XL, self.kscale,
                                         self.matches.obj_keypoints_2d, h)
        
        d2FdThetadXL = function_d2FdThetadXL_2(Theta, XL, self.kscale,
                                               self.matches.obj_keypoints_2d, h)
        
        inv_d2FdTheta2 = np.linalg.inv(d2FdTheta2)
        cov_pose = np.dot(inv_d2FdTheta2, 
                            np.dot(d2FdThetadXL.T, 
                                   np.dot(cov_XL, np.dot(d2FdThetadXL, 
                                                         inv_d2FdTheta2))))
        return cov_pose
    
    def get_scene(self, idx=None):
        if not idx is None:
            return (self._scene_,)[idx]
        else:
            return (self._scene_,)
    
    def set_detector_num_features(self, num_features):
        self.camera.set_detector_num_features(num_features)

def compute_pose_unc(image_points, mosaic_points, camera_matrix, wNm,
    pixel_noise_var, homography_matrix, camera_cov=None, wNm_cov=None, h=1e-6):
    """compute_pose_unc(image_points, mosaic_points, camera_matrix, wNm,
                        pixel_noise_var, homography_matrix, camera_cov=None,
                        wNm_cov=None, h=1e-6)
    where
        image_points -> Nx2 set of points in the camera image
        mosaic_points -> Nx2 set of corresponding points in the mosaic/template
        camera_matrix -> 3x3 camera matrix
        wNm -> Homography from world to mosaic/template (?)
        pixel_noise_var -> Noise variance for the pixel positions
        homography_matrix -> Initial homography from mosaic/template to image
        camera_cov -> 4x4 matrix for camera matrix uncertainty (default eye(4))
        wNm_cov -> 9x9 matrix of the covariance of wNm (default zeros((9, 9)))
        h -> Step size for optimisation
    """
    #import pose_uncertainty
    #import numpy as np
    if camera_cov is None:
        camera_cov = np.eye(4)
    
    if wNm_cov is None:
        wNm_cov = np.zeros((9, 9))
    
    # Normalise(?) camera matrix
    camera_matrix /= camera_matrix[2, 2]
    # Normalisation of camera matrix covariance (use Frobenius norm)
    kscale = 1./norm(camera_matrix) 
    kvec_scaled = camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]] * kscale
    
    assert np.abs(norm(wNm) - 1) < 1e-6, "Norm of wNm must be 1"
    
    # Initial homography matrix
    # where image_points = homography_matrix * mosaic_points
    Mimini, _CONDS, _A, _R = lsnormplanar(image_points, mosaic_points, 'stdv')
    MiWini = np.dot(Mimini, np.linalg.inv(wNm))
    
    # Generate initial pose parameters
    poseparams_ini = getposeparam_sturm(camera_matrix, MiWini, "WTC")
    
    num_points = image_points.shape[0]
    x_coord = (image_points*kscale)
    var_x_coord = (pixel_noise_var*kscale**2)*np.ones(num_points*2)
    
    XL = np.hstack((kvec_scaled, wNm.flatten(), x_coord.flatten()))
    
    fcost_ini = function_cost_total(poseparams_ini, XL, kscale, mosaic_points)
    print 'Initial cost: ', fcost_ini
    
    if not _minimize_ is None:
        result = _minimize_(function_cost_total, poseparams_ini,
                          (XL, kscale, mosaic_points),
                          method="L-BFGS-B", tol=1e-3)#"L-BFGS-B") #"Nelder-Mead")
    else:
        # Refine the pose using Simplex Downhill
        Tfin, fopt, num_iters, num_funcalls, warnflag = scipy.optimize.fmin(
            function_cost_total, poseparams_ini, (XL, kscale, mosaic_points),
            xtol=1e-6, ftol=1e-6, full_output=1)
        result = scipy.optimize.optimize.Result({"x":Tfin,
            "fun":fopt, "nfev":num_funcalls, "nit":num_iters,
            "success":not warnflag, "status":warnflag})
        
    Tfin = result.x
    fcost_fin = result.fun #function_cost_total(Tfin, XL, kscale, mosaic_points)
    print 'Initial cost: ', fcost_ini
    print 'Final cost:   ', fcost_fin
    
    Theta = Tfin
    POSEPAR = Tfin	# Devolver um vector linha de par�metros
    
    # Use a subset of the points to compute the covariance
#    max_num_points = np.round(num_points*0.125)
#    var_x_coord = (pixel_noise_var*kscale**2)*np.ones(max_num_points*2)
#    x_coord_subset_idx = np.unique(np.round(np.linspace(0, x_coord.shape[0]-1, max_num_points))).astype(np.int)
#    XL_subset = np.hstack((kvec_scaled, wNm.flatten(), x_coord[x_coord_subset_idx].flatten()))
    
    camera_cov_scaled = camera_cov * kscale**2
    # S� se est� a considerar ruido independente nas coordenadas
    cov_x_coor = np.diag(var_x_coord)
    # Adicionar covari�ncia dos par�metros intr�nsecos
    cov_XL = block_diag(camera_cov_scaled, wNm_cov, cov_x_coor)
    
    # Calculo da Hessiana de F em ordem a Tetha
    d2FdTheta2 = function_d2FdTheta2(Theta, XL, kscale, mosaic_points, h)
    
    # ==== Checked up to here ====
    
    d2FdThetadXL = function_d2FdThetadXL_2(Theta, XL, kscale, mosaic_points, h)
    
    #print "compute_pose_unc():123"
    #code.interact(local=locals())

    # Para debug: Calculo do Jacobiano da funcao que calcula explicitamente o 
    # Theta (getposeparam_sturm)
    #inv_h = 1./h
    #dThetadXL = np.zeros((XL.shape[0], Theta.shape[0]))
    #for n in range(XL.shape[0]):
    #    Theta1 = function_pose_from_XL(vadd(XL, n,  half_h), kscale, mosaic_points)
    #    Theta2 = function_pose_from_XL(vadd(XL, n, -half_h), kscale, mosaic_points)
    #    # troca de n e p em relacao a numxdiff.m
    #    dThetadXL[n, :] = inv_h * (Theta1 - Theta2)
    #
    #COVPOSE_dir = np.dot(dThetadXL.T, np.dot(cov_XL, dThetadXL))
    
    inv_d2FdTheta2 = np.linalg.inv(d2FdTheta2)
    COVPOSEPAR = np.dot(inv_d2FdTheta2, 
                        np.dot(d2FdThetadXL.T, 
                               np.dot(cov_XL, np.dot(d2FdThetadXL, 
                                                     inv_d2FdTheta2))))
    #print "compute_pose_unc():143"
    #code.interact(local=locals())
    
    return POSEPAR, COVPOSEPAR#, COVPOSE_dir
    
    
    
    
    
def function_cost_total(Thetain, XLin, kscale, COORm):
    """funcao de custo usando os parametros intrinsecos e a colineacao que 
    calibra metricamente o mosaico"""
    K = np.asarray([[XLin[0], 0, XLin[2]],
                    [0, XLin[1], XLin[3]],
                    [0, 0, kscale]])
    wNm = XLin[4:13].reshape((3, 3))
    TiW = pose1homo(Thetain, K, 'WTC')
    
    image_points = XLin[13:]/kscale
    num_points = image_points.shape[0]/2
    image_points = image_points.reshape((num_points, 2))
    assert 1 <= Thetain.ndim <= 2, ("Thetain must be 1D for single pose or \
        2D for mutiple poses")
    if Thetain.ndim == 1:
        PSI = np.dot(TiW, wNm)
        PSI /= norm(PSI, 'fro')
        
        COORm_em_i = trancoor(PSI, COORm)
        fcost = norm(image_points - COORm_em_i, 'fro')#/1000.
    else:
        PSI = np.asarray([np.dot(_TiW_, wNm) for _TiW_ in TiW])
        PSI_norm = np.asarray(map(norm, PSI))
        PSI /= PSI_norm[:, np.newaxis, np.newaxis]
        
        COORm_em_i = trancoor(PSI, COORm)
        fcost = np.asarray(map(norm, image_points - COORm_em_i))#/1000.
    
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


def vadd(vec_in, pos, amount):
    """vecout = vadd(VECIN, pos, amount)
    Add amount to vector VECIN in position pos
    """
    vec_out = vec_in.copy()
    assert 1 <= vec_in.ndim <= 2, "vec_in must be 1D for single vector or \
    2D for multiple vectors"
    if vec_in.ndim == 1:
        vec_out[pos] += amount
    else:
        vec_out[:, pos] += amount
    return vec_out


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
    
#    for n in range(num_theta):
#        poses_n = vadd(Theta_vec, n, half_h_vec1)
#        for p in range(num_theta):
#            #P1 = function_cost_total(vadd(vadd(Theta, n, half_h), p, half_h), XL, kscale, COORm)
#            #P2 = function_cost_total(vadd(vadd(Theta, n, half_h), p, -half_h), XL, kscale, COORm)
#            #P3 = function_cost_total(vadd(vadd(Theta, n, -half_h), p, half_h), XL, kscale, COORm)
#            #P4 = function_cost_total(vadd(vadd(Theta, n, -half_h), p, -half_h), XL, kscale, COORm)
#            #
#            ## troca de n e p em relacao a numhess.m
#            #P_sum = P1 -P2 -P3 +P4
#            #P_sign = sign(P_sum)
#            #log_d2FdTheta2[n, p] = log_inv_hsq + log(abs(P_sum))
#            #sign_d2FdTheta2[n, p] = P_sign
#            
#            poses_np = vadd(poses_n, p, half_h_vec2)
#            func_cost = function_cost_total(poses_np, XL, kscale, COORm)
#            
#            #poses[0, :] = vadd(vadd(Theta, n, half_h), p, half_h)
#            #poses[1, :] = vadd(vadd(Theta, n, half_h), p, -half_h)
#            #poses[2, :] = vadd(vadd(Theta, n, -half_h), p, half_h)
#            #poses[3, :] = vadd(vadd(Theta, n, -half_h), p, -half_h)
#            #func_cost = function_cost_total(poses, XL, kscale, COORm)
#            
#            P_sum = (func_cost*derivative_sign).sum()
#            P_sign = sign(P_sum)
#            log_d2FdTheta2[n, p] = log_inv_hsq + log(abs(P_sum))
#            sign_d2FdTheta2[n, p] = P_sign
#    d2FdTheta2 = np.exp(log_d2FdTheta2)*sign_d2FdTheta2
    return d2FdTheta2

def function_d2FdThetadXL_1(Theta, XL, kscale, COORm, h):
    # Calculo da segunda derivada de F em ordem a Tetha e PSI
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    #log_d2FdThetadXL = log_inv_hsq * np.ones((XL.shape[0], Theta.shape[0]))
    #sign_d2FdThetadXL = np.empty((XL.shape[0], Theta.shape[0]))
    log_d2FdThetadXL_all_p = log_inv_hsq * np.ones((XL.shape[0], Theta.shape[0]))
    sign_d2FdThetadXL_all_p = np.empty((XL.shape[0], Theta.shape[0]))
    half_h = h/2.
    
    #print "compute_pose_unc():100"
    #code.interact(local=locals())
    num_theta = Theta.shape[0]
    num_xl = XL.shape[0]
    half_h_delta = np.eye(num_theta)*half_h
    Theta_plus = np.repeat([Theta], num_theta, axis=0)
    Theta_minus = Theta_plus - half_h_delta
    Theta_plus += half_h_delta
    Theta_vec_all = np.hstack((Theta_plus, Theta_minus)).reshape((12, 6))
    XL_plus = XL.copy()
    XL_minus = XL.copy()
    for n in range(num_xl): #1:length(XL)
        old_XL_n = XL[n]
        XL_plus[n] += half_h
        XL_minus[n] -= half_h
        func_cost_p1_p2 = function_cost_total(Theta_vec_all, XL_plus, kscale, COORm).reshape((num_theta, 2))
        func_cost_p3_p4 = function_cost_total(Theta_vec_all, XL_minus, kscale, COORm).reshape((num_theta, 2))
        func_cost_p1_p2[:, 1] *= -1
        func_cost_p3_p4[:, 0] *= -1
        P_sum_all_p = (func_cost_p1_p2 + func_cost_p3_p4).sum(axis=1)
        P_sign_all_p = sign(P_sum_all_p)
        log_d2FdThetadXL_all_p[n] += log(abs(P_sum_all_p))
        sign_d2FdThetadXL_all_p[n] = sign(P_sign_all_p)
        XL_plus[n] = old_XL_n
        XL_minus[n] = old_XL_n
    d2FdThetadXL = sign_d2FdThetadXL_all_p*np.exp(log_d2FdThetadXL_all_p)
    
#    for n in range(num_xl):
#        for p in range(num_theta): #1:length(Theta)
#            P1 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n,  half_h) , kscale, COORm)
#            P2 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n,  half_h) , kscale, COORm)
#            P3 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n, -half_h), kscale, COORm)
#            P4 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n, -half_h), kscale, COORm)
#            #% troca de n e p em relacao a numxdiff.m
#            p1_p4 = P1 + P4
#            p2_p3 = P2 + P3
#            P_sum = p1_p4 - p2_p3
#            P_sign = sign(P_sum)
#            log_d2FdThetadXL[n, p] += log(abs(P_sum))
#            sign_d2FdThetadXL[n, p] = sign(P_sign)
#    
#    if not np.allclose(log_d2FdThetadXL, log_d2FdThetadXL_all_p):
#        print "Matrices differ"
#    d2FdThetadXL = sign_d2FdThetadXL*np.exp(log_d2FdThetadXL)
    return d2FdThetadXL


def function_d2FdThetadXL_2(Theta, XL, kscale, mosaic_points, h):
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
        #old_XL = XL[n]
        #XL_plus[n] += half_h
        #XL_minus[n] -= half_h
        XL_plus = vadd(XL, n, half_h)
        XL_minus = vadd(XL, n, -half_h)
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
        #XL_plus[n] = old_XL
        #XL_minus[n] = old_XL
    d2FdThetadXL = sign_d2FdThetadXL*np.exp(log_d2FdThetadXL)
    return d2FdThetadXL

def function_d2FdThetadXL_orig(Theta, XL, kscale, mosaic_points, h):
    # Calculo da segunda derivada de F em ordem a Tetha e PSI
    log_d2FdThetadXL = np.empty((XL.shape[0], Theta.shape[0]))
    sign_d2FdThetadXL = np.empty((XL.shape[0], Theta.shape[0]))
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    half_h = h/2.
    
    #print "compute_pose_unc():100"
    #code.interact(local=locals())
    
    for n in range(XL.shape[0]): #1:length(XL)
        for p in range(Theta.shape[0]): #1:length(Theta)
            #pose_vecs = np.asarray((vadd(Theta, p,  half_h), vadd(Theta, p, -half_h), vadd(Theta, p,  half_h), vadd(Theta, p, -half_h)))
            #xl_vecs = np.asarray((vadd(XL, n,  half_h), vadd(XL, n,  half_h), vadd(XL, n, -half_h), vadd(XL, n, -half_h)))
            P1 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n,  half_h), kscale, mosaic_points)
            P2 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n,  half_h), kscale, mosaic_points)
            P3 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n, -half_h), kscale, mosaic_points)
            P4 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n, -half_h), kscale, mosaic_points)
            #% troca de n e p em relacao a numxdiff.m
            p1_p4 = P1 + P4
            p2_p3 = P2 + P3
            P_sum = p1_p4 - p2_p3
            P_sign = sign(P_sum)
            log_d2FdThetadXL[n, p] = log_inv_hsq + log(abs(P_sum))
            sign_d2FdThetadXL[n, p] = sign(P_sign)
    
    d2FdThetadXL = sign_d2FdThetadXL*np.exp(log_d2FdThetadXL)
    return d2FdThetadXL

def function_pose_from_XL(XLin, kscale, mosaic_points):
    camera_matrix = np.asarray([[XLin[0], 0, XLin[2]],
                                [0, XLin[1], XLin[3]],
                                [0, 0, kscale]])
    wNm = np.reshape(XLin[4:13], (3, 3))
    
    image_points = XLin[13:].copy()
    num_points = image_points.shape[0]/2
    image_points = image_points.reshape((num_points, 2))/kscale
    
    PSImat, _CONDS, _A, _R = lsnormplanar(image_points, mosaic_points, 'stdv')
    Thetaout = getposeparam_sturm(camera_matrix,
                                  np.dot(PSImat, np.linalg.inv(wNm)), 'WTC')
    return Thetaout



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


test_data = scipy.io.loadmat("/home/snagappa/udg/ros/udg_pandora/src/pose_cov/test_data.mat")
coori = test_data["coori"]
coorm = test_data["coorm"]
camera_matrix = test_data["K"]
wNm = test_data["MWR_norm"]
pixel_noise_var = test_data["sigmai"][0, 0]
homography_matrix = None
camera_cov = test_data["KCOV"]
wNm_cov = test_data["COVMWR"]

print "result = objdetect.compute_pose_unc(coori, coorm, camera_matrix, wNm, pixel_noise_var, homography_matrix, camera_cov, wNm_cov)"

if __name__ == "__main__":
    result = compute_pose_unc(coori, coorm, camera_matrix, wNm, pixel_noise_var, homography_matrix, camera_cov, wNm_cov)
    print "\nPose:"
    print result[0]
    print "\nPose Covariance:"
    print np.diag(result[1])
    
    print "\nAbs Pose Error:"
    print np.abs(result[0] - test_data["POSEPAR"])
    print "\nAbs Pose Cov Error:"
    print np.diag(np.abs(result[1] - test_data["COVPOSEPAR"]))
