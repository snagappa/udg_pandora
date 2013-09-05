# -*- coding: utf-8 -*-
#cython: wraparound=False, boundscheck=False, profile=True

# ROS imports
import roslib
roslib.load_manifest("udg_pandora")
import rospy
from tf import LookupException, ExtrapolationException
# Python imports
import numpy as np
cimport numpy as np
from cpython cimport bool
import sys, traceback
from collections import namedtuple

# Custom imports
import blas
import cameramodels
from kalmanfilter import kf_update_cov, kf_update_x, ukf_update_cov, sigma_pts
from misctools import STRUCT
from rfs_merge import rfs_merge
import code

DEBUG = True
blas.SET_DEBUG(False)

SAMPLE = namedtuple("SAMPLE", "weight state covariance")


###############################################################################
###############################################################################

    
tiny = np.finfo(np.double).tiny
log_tiny = np.log(tiny)

#def phd_update(weights, states, covs, timestamp, filter_vars, camera, observations,
#               observation_noise, USE_3D, USE_UKF, prune_threshold=0,
#               DISPLAY=False):
def phd_update(np.ndarray[double, ndim=1] weights not None,
               np.ndarray[double, ndim=2] states not None,
               np.ndarray[double, ndim=3] covs not None, timestamp,
               filter_vars, camera,
               np.ndarray[double, ndim=2] observations not None,
               np.ndarray[double, ndim=3] observation_noise not None,
               USE_3D, USE_UKF, prune_threshold=0, DISPLAY=None):
    """phd.update(observations, observation_noise)
    Update the current landmarks using the new observations
    observations - numpy array of size Nx3
    observation_noise - numpy array of size Nx3x3
    """
    
    cdef int num_observations, z_dim, obs_count
    cdef double normalisation_factor
    cdef np.ndarray[double, ndim=1] detection_probability
    cdef np.ndarray[double, ndim=1] clutter_pdf, clutter_intensity
    cdef np.ndarray[double, ndim=1] prev_weights, detected_weights
    cdef np.ndarray[double, ndim=2] prev_states, detected_states, pred_z
    cdef np.ndarray[double, ndim=3] prev_covs, detected_covs, h_mat
    cdef np.ndarray[double, ndim=1] _observation_, x_pdf
    cdef np.ndarray[double, ndim=2] this_observation_noise, residuals
    cdef np.ndarray[double, ndim=1] this_updated_weights, updated_weights
    cdef np.ndarray[double, ndim=2] this_updated_states, updated_states
    cdef np.ndarray[double, ndim=3] this_updated_covs, updated_covs
    cdef list updated_weights_list, updated_states_list, updated_covs_list
    
    num_observations = (observations.shape[0] 
                        if ( not observations.shape[1]==0) else 0)
    z_dim = observations.shape[1]
    
    if not weights.shape[0]:
        #print "nothing to update"
        return weights, states, covs
    
    detection_probability = camera.pdf_detection(states, timestamp, margin=0)
    
    try:
        # Account for missed detection
        prev_weights = weights.copy()
        prev_states = states.copy()
        prev_covs = covs.copy()
        updated_weights_list = [weights.copy()*(1-detection_probability)]
        updated_states_list = [states.copy()]
        updated_covs_list = [covs.copy()]
        
        # Do the update only for detected landmarks
        detected_indices = detection_probability > 0
        detected_weights = ( prev_weights[detected_indices]*
                             detection_probability[detected_indices] )
        detected_states = prev_states[detected_indices]
        detected_covs = prev_covs[detected_indices]
        
        if detected_weights.shape[0]:
            # Covariance update part of the Kalman update is common to all 
            # observation-updates
            if num_observations:
                if USE_3D:
                    # Observations will appear at position given by opposite
                    # rotation of the parent
                    # With 3D point observations
                    if not USE_UKF:
                        h_mat = np.asarray(
                            camera.observation_jacobian(timestamp)[np.newaxis],
                            order='C')
                        pred_z = camera.observations(detected_states,
                                                     timestamp)[0]
                    clutter_pdf = camera.pdf_clutter(observations, timestamp)
                    clutter_intensity = (
                        filter_vars.clutter_intensity*clutter_pdf)
                else:
                    # With pixel observations
                    if not USE_UKF:
                        h_mat = camera.observation_px_jacobian_full(
                            detected_states, timestamp)
                        pred_z = camera.observations_px(detected_states,
                                                        timestamp)
                    
                    clutter_pdf = camera.pdf_clutter_disparity(observations)
                    clutter_intensity = (
                        filter_vars.clutter_intensity*clutter_pdf)
                
                # Do a single covariance update if equal observation noise
                this_observation_noise = observation_noise[0]
                if not USE_UKF:
                    this_updated_covs, kalman_info = kf_update_cov(
                        detected_covs, h_mat, this_observation_noise,
                        INPLACE=False)
                else:
                    if USE_3D:
                        obsfn = camera.observations
                        obsfn_args = (timestamp, 1,)
                    else:
                        obsfn = camera.observations_px
                        obsfn_args = (timestamp,)
                    try:
                        (this_updated_covs, pred_z, 
                         kalman_info) = ukf_update_cov(
                            detected_states, detected_covs,
                            this_observation_noise, obsfn, obsfn_args,
                            _alpha=1e-3, _beta=2, _kappa=0, INPLACE=False)
                    except:
                        code.interact(local=locals())
            # We need to update the states and find the updated weights
            for obs_count in range(num_observations):
                _observation_ = observations[obs_count]
                # Apply the Kalman update to get the new state
                this_updated_states, residuals = kf_update_x(
                    detected_states, pred_z, _observation_, 
                    kalman_info.kalman_gain, INPLACE=False)
                # Calculate the weight of the Gaussians for this observation
                # Calculate term in the exponent
                x_pdf = np.exp(-0.5*np.power(
                    blas.dgemv(kalman_info.inv_sqrt_S, 
                               residuals, TRANSPOSE_A=True), 2).sum(axis=1))/ \
                    np.sqrt(kalman_info.det_S*(2*np.pi)**z_dim)
                #x_pdf = misctools.mvnpdf(_observation_, pred_z, kalman_info.S)
                this_updated_weights = detected_weights*x_pdf
                # Normalise the weights
                normalisation_factor = ( clutter_intensity[obs_count] + 
                                         #self.vars.birth_intensity +
                                         this_updated_weights.sum() )
                this_updated_weights /= normalisation_factor
                
                # Create new state with new_x and P to add to _states_
                valid_idx = this_updated_weights > prune_threshold
                if np.count_nonzero(valid_idx):
                    updated_weights_list += [this_updated_weights[valid_idx]]
                    updated_states_list += [this_updated_states[valid_idx]]
                    updated_covs_list += [this_updated_covs[valid_idx]]
                
                if DISPLAY == True:
                    print "\nUpdated weights:"
                    print this_updated_weights[valid_idx]
        
        updated_weights = np.concatenate(updated_weights_list)
        updated_states = np.concatenate(updated_states_list)
        updated_covs = np.concatenate(updated_covs_list)
        
    except:
        print "error in update"
        exc_info = sys.exc_info()
        print "GMPHD:UPDATE():\n", traceback.print_tb(exc_info[2])
        raise
    return updated_weights, updated_states, updated_covs

#def phd_prune(weights, states, covs,
#              prune_threshold=-1, max_num_components=-1):
def phd_prune(np.ndarray[double, ndim=1] weights,
              np.ndarray[double, ndim=2] states,
              np.ndarray[double, ndim=3] covs,
              double prune_threshold=-1, int max_num_components=-1):
    """phd.prune()
    Remove landmarks in the map with low weights.
    """
    
    cdef int num_components = weights.shape[0]
    
    num_components = weights.shape[0]
    if max_num_components < 0:
        max_num_components = 65535
    
    try:
        # Check if we need to prune
        if ((not num_components) or 
            ((prune_threshold <= 0) and 
             (num_components < max_num_components))):
            return weights, states, covs
        
        if prune_threshold > 0:
            valid_idx = weights >= prune_threshold
            weights = weights[valid_idx]
            states = states[valid_idx]
            covs = covs[valid_idx]
        
        # Prune to the maximum number of components
        if ((max_num_components > 0) and 
            (num_components > max_num_components)):
            print "Hit maximum number of components, pruning..."
            # Sort in ascending order and 
            # select the last max_num_components states
            idx = weights.argsort()[-max_num_components:]
            weights = weights[idx]
            states = states[idx]
            covs = covs[idx]
        return weights, states, covs
    except:
        exc_info = sys.exc_info()
        print "GMPHD:PRUNE():\n", traceback.print_tb(exc_info[2])
        raise

#def camera_birth_disparity(camera, birth_intensity, features_rel, features_cv,
#                           timestamp=None):
def camera_birth_disparity(camera, double birth_intensity,
                           np.ndarray[double, ndim=2] features_rel not None,
                           np.ndarray[double,ndim=3] features_cv not None,
                           timestamp=None):
    """phd.camera_birth(parent_ned, parent_rpy, features_rel, 
        features_cv=None) -> birth_weights, birth_states, birth_covariances
    Create birth components using features visible from the camera.
    parent_ned - numpy array of parent position (north, east, down)
    parent_rpy - numpy array of parent orientation (roll, pitch, yaw)
    features_rel - Nx3 numpy array of feature positions relative to parent
    features_cv - Nx3x3 numpy array of covariance of the features
    """
    
    cdef np.ndarray[double, ndim=1] birth_wt, x_weight, p_weight, cov_wts
    cdef np.ndarray[double, ndim=2] birth_st
    cdef np.ndarray[double, ndim=3] birth_cv, zz_sigma_cov
    cdef np.ndarray[double, ndim=2] birth_features
    cdef np.ndarray[double, ndim=2] img_pts_l, img_pts_r
    cdef np.ndarray[double, ndim=3] x_sigma, z_sigma
    cdef np.ndarray[double, ndim=2] x_sigma_flat, z_sigma_flat, z_diff_flat
    cdef int num_features, num_states, state_dim
    
    birth_wt = np.empty(0)
    birth_st = np.empty((0, 3))
    birth_cv = np.empty((0, 3, 3))
#    base_cov = [0.005, 0.005, 0.01]
    num_features = features_rel.shape[0] if (not features_rel.shape[1]==0) else 0
    
    if num_features:
        num_states = features_rel.shape[0]
        state_dim = features_rel.shape[1]
        num_sigma_pts = 2*state_dim + 1
        # Draw sigma points from each of the disparity points
        x_sigma, x_weight, p_weight = sigma_pts(
            features_rel, features_cv, _alpha=0.5)
        
        # Generate observations for predicted sigma points
        x_sigma_flat = np.reshape(x_sigma, (num_states*num_sigma_pts,
                                            x_sigma.shape[2]))
        
        img_pts_l = x_sigma_flat[:, :2]
        img_pts_r = x_sigma_flat[:, :2].copy()
        img_pts_r[:, 0] -= x_sigma_flat[:, 2]
        
        # Convert the sigma points from disparity space to 3D space
        z_sigma_flat = camera.triangulate(img_pts_l, img_pts_r)
        try:
            z_sigma_flat = camera.to_world_coords(z_sigma_flat, timestamp,
                                              rethrow_exception=True)
        except (LookupException, ExtrapolationException) as exc_msg:
            print "Could not compute tranformation, ignoring timestamp and trying again"
            print exc_msg
            z_sigma_flat = camera.to_world_coords(z_sigma_flat,
                                                  rethrow_exception=False)
        if z_sigma_flat.shape[0]:
            z_dim = z_sigma_flat.shape[1]
        else:
            print "Failed to generate birth terms. Aborting birth."
            return birth_wt, birth_st, birth_cv
        
        z_sigma = np.reshape(z_sigma_flat, (num_states, num_sigma_pts, z_dim))
        
        predicted_z = (x_weight[np.newaxis, :, np.newaxis]*z_sigma).sum(axis=1)
        
        # Observation covariance
        z_diff_flat = np.asarray(
            np.reshape(z_sigma - predicted_z[:, np.newaxis, :],
            (num_states*num_sigma_pts, z_dim)), order='C')
        
        # Innovation covariance - used to compute the likelihood
        cov_wts = (p_weight * np.ones((num_states, 1))).ravel()
        zz_sigma_cov = np.asarray(
            np.reshape(blas.dsyr('l', z_diff_flat, cov_wts),
                       (num_states, num_sigma_pts, z_dim, z_dim)).sum(axis=1),
                        order='C')
        blas.symmetrise(zz_sigma_cov, 'l')
        
        birth_features = predicted_z
        if birth_features.shape[0]:
            try:
                birth_wt = birth_intensity*np.ones(birth_features.shape[0])
                birth_st = predicted_z
                birth_cv = zz_sigma_cov
            except:
                print "tf conversion to world coords failed!"
                exc_info = sys.exc_info()
                print "GMPHD:CAMERA_BIRTH():\n", traceback.print_tb(exc_info[2])
                raise
    return (birth_wt, birth_st, birth_cv)

#def camera_birth(camera, birth_intensity, features_rel, features_cv,
#                 timestamp=None):
def camera_birth(camera, double birth_intensity,
                 np.ndarray[double, ndim=2] features_rel not None, 
                 np.ndarray[double, ndim=3] features_cv not None,
                 timestamp=None):
    """phd.camera_birth(parent_ned, parent_rpy, features_rel, 
        features_cv=None) -> birth_weights, birth_states, birth_covariances
    Create birth components using features visible from the camera.
    parent_ned - numpy array of parent position (north, east, down)
    parent_rpy - numpy array of parent orientation (roll, pitch, yaw)
    features_rel - Nx3 numpy array of feature positions relative to parent
    features_cv - Nx3x3 numpy array of covariance of the features
    """
    
    cdef np.ndarray[double, ndim=1] birth_wt
    cdef np.ndarray[double, ndim=2] birth_st
    cdef np.ndarray[double, ndim=3] birth_cv
    cdef np.ndarray[double, ndim=2] birth_features
    cdef int num_features
    
    birth_wt = np.empty(0)
    birth_st = np.empty((0, 3))
    birth_cv = np.empty((0, 3, 3))
    num_features = (features_rel.shape[0]
                    if (not features_rel.shape[1]==0) else 0)
    
    if num_features:
        #covs = ([0.01, 0.01, 0.01]*np.eye(3)[np.newaxis])*np.ones((num_features, 1, 1))
        covs = features_cv[0]*np.ones((num_features, 1, 1))
        
        birth_features = features_rel
        if birth_features.shape[0]:
            try:
                birth_st = camera.to_world_coords(birth_features,
                    timestamp, rethrow_exception=True)
            except (LookupException, ExtrapolationException) as exc_msg:
                print "Could not compute tranformation, ignoring timestamp and trying again"
                print exc_msg
                birth_st = camera.to_world_coords(birth_features,
                    rethrow_exception=True)
                if not birth_st.shape[0]:
                    print "Failed to generate birth terms. Aborting birth."
                    return birth_wt, birth_st, birth_cv
            else:
                birth_wt = birth_intensity*np.ones(birth_features.shape[0])
                birth_cv = covs
    return (birth_wt, birth_st, birth_cv)

#def merge_fov(camera, weights, states, covs, merge_threshold,
#              timestamp=None, detection_threshold=0.3):
def merge_fov(camera, np.ndarray[double, ndim=1] weights not None,
              np.ndarray[double, ndim=2] states not None,
              np.ndarray[double, ndim=3] covs not None,
              double merge_threshold, timestamp=None,
              double detection_threshold=0.3):
    """phd.merge_fov(detection_threshold=0.5)
    Merge Gaussian components which are in the field of view or which
    satisfy a probability of detection given by detection_threshold.
    Set detection_threshold to 0 to merge landmarks everywhere in the map.
    """
    
    cdef np.ndarray[double, ndim=1] merged_wts, unmerged_wts
    cdef np.ndarray[double, ndim=2] merged_sts, unmerged_sts
    cdef np.ndarray[double, ndim=3] merged_cvs, unmerged_cvs
    
    try:
        if (merge_threshold < 0) or (weights.shape[0] < 2):
            return weights, states, covs
        # Get objects in the field of view
        if detection_threshold <= 0:
            undetected_idx = np.empty(0, dtype=np.int)
            detected_idx = np.arange(states.shape[0])
        else:
            # Convert states to camera coordinate system
            try:
                # Stereo camera - get relative position for left and right
                detected_idx = camera.pdf_detection(states, timestamp)
            except:
                exc_info = sys.exc_info()
                print "Error merging states"
                print "GMPHD:MERGE_FOV():\n", traceback.print_tb(exc_info[2])
                raise
            undetected_idx = (detected_idx < detection_threshold)
            detected_idx = (detected_idx >= detection_threshold)
        #undetected_idx = misctools.gen_retain_idx(self.weights.shape[0], 
        #                                          detected_idx)
        
        if not detected_idx.shape[0]:
            return weights, states, covs
        
        # Save samples which are not going to be merged
        unmerged_wts = weights[undetected_idx]
        unmerged_sts = states[undetected_idx]
        unmerged_cvs = covs[undetected_idx]
        # Remove unmerged samples from the state
        weights = weights[detected_idx]
        states = states[detected_idx]
        covs = covs[detected_idx]
        #num_remaining_components = self.weights.shape[0]
        
        merged_wts, merged_sts, merged_cvs = rfs_merge(weights, states, 
                                                       covs, merge_threshold)
        merged_wts = np.hstack((unmerged_wts, merged_wts))
        merged_sts = np.vstack((unmerged_sts, merged_sts))
        merged_cvs = np.vstack((unmerged_cvs, merged_cvs))
        return (merged_wts, merged_sts, merged_cvs)
    except:
        exc_info = sys.exc_info()
        print "GMPHD:MERGE_FOV():\n", traceback.print_tb(exc_info[2])
        raise

#def phd_iterate(weights, states, covs, timestamp, filter_vars, camera,
#                observations, obs_noise, USE_3D, USE_UKF=False, DISPLAY=False):
def phd_iterate(np.ndarray[double, ndim=1] weights,
                np.ndarray[double, ndim=2] states,
                np.ndarray[double, ndim=3] covs, timestamp, filter_vars,
                camera, np.ndarray[double, ndim=2] observations,
                np.ndarray[double, ndim=3] obs_noise, USE_3D, USE_UKF=False,
                DISPLAY=False):
    """phd.iterate(observations, obs_noise)
    Perform a single iteration of the filter:
        predict()
        update()
        merge()
        prune()
    """
    
    cdef np.ndarray[double, ndim=1] updated_weights, birth_wt
    cdef np.ndarray[double, ndim=2] updated_states, birth_st
    cdef np.ndarray[double, ndim=3] updated_covs, birth_cv
    
    # Birth
    if USE_3D:
        (birth_wt, birth_st, birth_cv) = camera_birth(
            camera, filter_vars.birth_intensity, observations, obs_noise,
            timestamp)
    else:
        (birth_wt, birth_st, birth_cv) = camera_birth_disparity(
            camera, filter_vars.birth_intensity, observations, obs_noise,
            timestamp)
    
    # Update
    updated_weights, updated_states, updated_covs = (
        phd_update(weights, states, covs, timestamp, filter_vars, camera,
                   observations, obs_noise, USE_3D, USE_UKF,
                   filter_vars.prune_threshold, DISPLAY))
    # Prune
    updated_weights, updated_states, updated_covs = phd_prune(
        updated_weights, updated_states, updated_covs,
        filter_vars.prune_threshold, max_num_components=65535)
    if updated_weights.shape[0] > filter_vars.max_num_components:
        # Merge over entire map
        updated_weights, updated_states, updated_covs = merge_fov(
            camera, updated_weights, updated_states, updated_covs,
            filter_vars.merge_threshold, detection_threshold=-1)
        updated_weights, updated_states, updated_covs = phd_prune(
            updated_weights, updated_states, updated_covs,
            filter_vars.prune_threshold, filter_vars.max_num_components)
    
    if birth_wt.shape[0]:
        updated_weights = np.hstack((updated_weights, birth_wt))
        updated_states = np.vstack((updated_states, birth_st))
        updated_covs = np.vstack((updated_covs, birth_cv))
    # Merge over field of view
    updated_weights, updated_states, updated_covs = merge_fov(
        camera, updated_weights, updated_states, updated_covs,
        filter_vars.merge_threshold, timestamp=timestamp,
        detection_threshold=1e-6)
    return updated_weights, updated_states, updated_covs

#def phd_estimate(weights, states, covs, USE_INTENSITY=True):
def phd_estimate(np.ndarray[double, ndim=1] weights,
                 np.ndarray[double, ndim=2] states,
                 np.ndarray[double, ndim=3] covs, bool USE_INTENSITY=True):
    """phd.estimate -> (weights, states, covariances)
    Estimate the number of landmarks in the map and return as a tuple of
    the weights, states and covariances of the landmarks.
    """
    cdef double intensity, num_targets_soft, num_targets
    
    if USE_INTENSITY:
        intensity = weights.sum()
        num_targets_soft = min((intensity, weights.shape[0]))
        num_targets = int(round(num_targets_soft))
    else:
        valid_targets = weights>0.5
        num_targets = valid_targets.sum()
    
    if num_targets:
        if USE_INTENSITY:
            inds = np.flipud(weights.argsort())
            inds = inds[0:num_targets]
            est_weights = weights[inds]
            est_states = states[inds]
            est_covs = covs[inds]
            # Discard states with low weight
            #valid_idx = np.where(est_weights>0.4)[0]
            #self._estimate_ = SAMPLE(est_weights[valid_idx],
            #                         est_states[valid_idx], 
            #                         est_covs[valid_idx])
        else:
            est_weights = weights[valid_targets]
            est_states = states[valid_targets]
            est_covs = covs[valid_targets]
        _estimate_ = SAMPLE(est_weights, est_states, est_covs)
    else:
        _estimate_ = SAMPLE(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3)))
    return _estimate_


class GMPHD(object):
    def __init__(self, filter_vars):
        self.map = STRUCT()
        self.map.weights = np.empty(0)
        self.map.states = np.empty((0, 3))
        self.map.covs = np.empty((0, 3, 3))
        self.camera = cameramodels.StereoCameraModel()
        self.map_estimate = None
        self.filter_vars = filter_vars
        self.estimate()
    
    def fromCameraInfo(self, camera_info_left, camera_info_right):
        self.camera.fromCameraInfo(camera_info_left, camera_info_right)
        left_tf_frame = camera_info_left.header.frame_id
        right_tf_frame = left_tf_frame+"_right"
        self.camera.set_tf_frame(left_tf_frame, right_tf_frame)
        fov_far = rospy.get_param("slam_feature_detector/fov/far")
        self.camera.set_near_far_fov(fov_far=fov_far)
    
#    def update_features(self, features, timestamp=None, USE_3D=False,
#                        USE_UKF=True, DISPLAY=False):
    def update_features(self, np.ndarray[double, ndim=2] features not None,
                        timestamp=None, USE_3D=False, USE_UKF=True,
                        DISPLAY=False):
        
        cdef np.ndarray[double, ndim=1] weights
        cdef np.ndarray[double, ndim=2] states
        cdef np.ndarray[double, ndim=3] covs
        
        if features.shape[0]:
            features_pos = features[:, :3].copy()
            features_noise = np.array([np.diag(features[i, 3:6]) 
                for i in range(features.shape[0])])
        else:
            features_pos = np.empty((0, 3))
            features_noise = np.empty((0, 3, 3))
        phd_map = self.map
        weights = phd_map.weights
        states = phd_map.states
        covs = phd_map.covs
        weights, states, covs = phd_iterate(weights, states, covs, timestamp,
            self.filter_vars, self.camera, features_pos, features_noise,
            USE_3D, USE_UKF, DISPLAY)
        self.map.weights = weights
        self.map.states = states
        self.map.covs = covs
    
    def map_to_image_points(self, timestamp=None):
        camera = self.camera
        weights = self.map.weights
        states = self.map.states
        
        is_visible = camera.is_visible(states, timestamp)
        is_visible[weights < 0.4] = False
        if is_visible.shape[0]:
            camera_points = camera.from_world_coords(states[is_visible], timestamp)
            img_points = camera.project3dToPixel(camera_points[0])
        else:
            img_points = (np.empty(0), np.empty(0))
        #print "Generated image points"
        return img_points
    
    def estimate(self):
        """estimate(self) -> estimate
        Generate the state and map estimates
        """
        #cam_position = self.camera.to_world_coords(np.zeros((1, 3)))
        self.map_estimate = phd_estimate(self.map.weights,
                self.map.states, self.map.covs)
        return self.map_estimate
    
