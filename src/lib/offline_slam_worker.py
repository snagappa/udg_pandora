# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:15:54 2012

@author: snagappa
"""


USE_CYTHON = False


import numpy as np
import misctools, blas
#from featuredetector import sensors, tf
import cameramodels
import local_tf as tf
cameramodels.tf = tf
from kalmanfilter import np_kf_predict_cov, kf_predict_cov, np_kf_update_cov, kf_update_cov, np_kf_update_x, kf_update_x
from collections import namedtuple
import copy
import code
from misctools import STRUCT, rotation_matrix
import sys
import traceback
from joblib import Parallel, delayed
from lib.rfs_merge import rfs_merge
import rospy
#for plotting
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

DEBUG = True
blas.SET_DEBUG(False)

### CYTHON STUFF ###
if USE_CYTHON:
    exec """
    import cython
    cimport numpy as np
    
    @cython.boundscheck(False)
    @cython.wraparound(False)"""


SAMPLE = namedtuple("SAMPLE", "weight state covariance")
    
class GMPHD(object):
    def __init__(self):
        """GMPHD() -> phd
        
        Backend class for the Gaussian Mixture PHD SLAM algorithm.
        This is based on formulation of SLAM as a single cluster process. The
        cluster centre is the parent's (vehicle's) pose and the daughter
        process represents the landmarks.
        
        """
        self.weights = np.zeros(0)
        self.states = np.zeros((0, 3))
        self.covs = np.zeros((0, 3, 3))
        self.parent_ned = np.zeros(3)
        self.parent_rpy = np.zeros(3)
        self.slam_info = STRUCT()
        self.slam_info.likelihood = 1.
        
        self._estimate_ = SAMPLE(np.zeros(0), 
                                 np.zeros((0, 3)), np.zeros((0, 3, 3)))
        
        self.vars = STRUCT()
        # Prune components less than this weight
        config_root = "phdslam/vars/"
        self.vars.prune_threshold = (
            rospy.get_param(config_root+"prune_threshold"))
        # Merge components  closer than this threshold
        self.vars.merge_threshold = (
            rospy.get_param(config_root+"merge_threshold"))
        # Maximum number of components to track
        self.vars.max_num_components = (
            rospy.get_param(config_root+"max_num_components"))
        # Intensity of new targets
        self.vars.birth_intensity = (
            rospy.get_param(config_root+"birth_intensity"))
        # Intensity of clutter in the scene
        self.vars.clutter_intensity = (
            rospy.get_param(config_root+"clutter_intensity"))
        # Probability of detection of targets in the FoV
        self.vars.pd = rospy.get_param(config_root+"pd")
        #self.vars.far_fov = 5.0
        #self.vars.ps = 1.0
        
        # Temporary variables to speed up some processing across different functions
        self.tmp = STRUCT()
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = False
        
        self.sensors = STRUCT()
        try:
            self.sensors.camera = cameramodels.StereoCameraModel()
        except:
            print "Error initialising camera models"
            exc_info = sys.exc_info()
            print "GMPHD:__INIT__():\n", traceback.print_tb(exc_info[2])
            raise
            
        self.sensors.camera.set_const_pd(self.vars.pd)
    
    def copy(self):
        """phd.copy() -> phd_copy
        Create a new copy of the GMPHD object"""
        new_object = GMPHD()
        try:
            new_object.weights = self.weights.copy()
            new_object.states = self.states.copy()
            new_object.covs = self.covs.copy()
            new_object.parent_ned = self.parent_ned.copy()
            new_object.parent_rpy = self.parent_rpy.copy()
            new_object._estimate_ = self.estimate()
            new_object.vars = copy.copy(self.vars)
            new_object.flags = copy.copy(self.flags)
            new_object.sensors.camera = self.sensors.camera.copy()
        except:
            exc_info = sys.exc_info()
            print "GMPHD:COPY():\n", traceback.print_tb(exc_info[2])
            raise
        return new_object
    
    def set_states(self, ptr_weights, ptr_states, ptr_covs):
        """phd.set_states(ptr_weights, ptr_states, ptr_covs)
        Assign new weights, states and covariances to the object. A copy of the
        arguments is not made - changes to the contents of the original
        variables in-place will be visible inside the object.
        """
        try:
            self.flags.ESTIMATE_IS_VALID = False
            self.weights = ptr_weights
            self.states = ptr_states
            self.covs = ptr_covs
        except:
            exc_info = sys.exc_info()
            print "GMPHD:SET_STATES():\n", traceback.print_tb(exc_info[2])
            raise
    
    def set_parent(self, parent_ned, parent_rpy):
        """phd.set_parent(parend_ned, parent_rpy)
        Set new values of the parent state
        parent_ned - numpy array indicating position as (north, east, down)
        parent_rpy - numpy array indicating orientation as (roll, pitch, yaw)
        """
        try:
            self.parent_ned = parent_ned.copy()
            self.parent_rpy = parent_rpy.copy()
        except:
            exc_info = sys.exc_info()
            print "GMPHD:SET_PARENT():\n", traceback.print_tb(exc_info[2])
            raise
    
    def birth(self, features_rel, features_cv=None, APPEND=False):
        """phd.birth(features_rel, features_cv=None, APPEND=False) 
        -> birth_wt, birth_st, birth_cv
        Generate a Gaussian mixture for the birth of new targets.
        features_rel -  numpy array of size Nx3 indicating relative position
        of features/landmarks with respect to the parent state
        features_cv - numpy array of size (Nx3x3)
        APPEND - will append the birth components to the current state if True
        """
        b_wt, b_st, b_cv = self.camera_birth(self.parent_ned, self.parent_rpy, 
                                             features_rel, features_cv)
        if APPEND and b_wt.shape[0]:
            self.flags.ESTIMATE_IS_VALID = False
            self.append(b_wt, b_st, b_cv)
        return (b_wt, b_st, b_cv)
    
    def predict(self, *args, **kwargs):
        """phd.predict(*args, **kwargs)
        Perform a prediction on the current landmarks. Since the landmarks are
        stationary, this does not do anything. 
        """
        pass
        """
        if self.covs.shape[0] == 0:
            return
        self.flags.ESTIMATE_IS_VALID = False
        # Get states which are in the field of view
        visible_idx = self.sensors.camera.is_visible(self.states, margin=0)
        extra_cov = np.eye(self.states.shape[1])*1e-5
        self.covs[visible_idx] += extra_cov
        """
        #if self.covs.shape[0]:
            #extra_cov = 0.005*np.eye(3)
            #self.covs += extra_cov[np.newaxis, :, :]
            #self.covs += np.eye(3)*self.covs*0.1
        #pass
    
    def update(self, observations, observation_noise):
        """phd.update(observations, observation_noise)
        Update the current landmarks using the new observations
        observations - numpy array of size Nx3
        observation_noise - numpy array of size Nx3x3
        """
        self.flags.ESTIMATE_IS_VALID = False
        # Container for slam parent update
        self.slam_info.log_likelihood = 1.
        self.slam_info.likelihood = 1.
        num_observations, z_dim = (observations.shape + (3,))[0:2]
        
        if not self.weights.shape[0]:
            #print "nothing to update"
            return self.slam_info
        camera = self.sensors.camera
        detection_probability = camera.pdf_detection(self.states, margin=0)
        clutter_pdf = camera.pdf_clutter(observations)
        clutter_intensity = self.vars.clutter_intensity*clutter_pdf
        
        try:
            # Account for missed detection
            prev_weights = self.weights.copy()
            prev_states = self.states.copy()
            prev_covs = self.covs.copy()
        
            updated = STRUCT()
            updated.weights = [self.weights*(1-detection_probability)]
            updated.states = [self.states]
            updated.covs = [self.covs]
            
            # Do the update only for detected landmarks
            detected_indices = detection_probability > 0
            detected = STRUCT()
            detected.weights = ( prev_weights[detected_indices]*
                                 detection_probability[detected_indices] )
            detected.states = prev_states[detected_indices]
            detected.covs = prev_covs[detected_indices]
            #ZZ SLAM,  step 1:
            self.slam_info.exp_sum__pd_predwt = np.exp(-detected.weights.sum())
            
            # SLAM, prep for step 2:
            self.slam_info.sum__clutter_with_pd_updwt = np.zeros(num_observations)
            
            if detected.weights.shape[0]:
                # Covariance update part of the Kalman update is common to all 
                # observation-updates
                if observations.shape[0]:
                    # Observations will appear at position given by opposite
                    # rotation of the parent
                    h_mat = np.asarray(
                        self.sensors.camera.observation_jacobian()[np.newaxis], 
                        order='C')
                    pred_z = self.sensors.camera.observations(detected.states)[0]
                    # Do a single covariance update if equal observation noise
                    if np.all(observation_noise == observation_noise[0]):
                        EQUAL_OBS_NOISE = True
                        this_observation_noise = observation_noise[0]
                        _this_detected_covs_, kalman_info = kf_update_cov(
                            detected.covs, h_mat, this_observation_noise,
                            INPLACE=False)
                    else:
                        EQUAL_OBS_NOISE = False
                # We need to update the states and find the updated weights
                for (_observation_, obs_count) in zip(observations, 
                                                      range(num_observations)):
                    if not EQUAL_OBS_NOISE:
                        this_observation_noise = observation_noise[obs_count]
                        this_detected_covs, kalman_info = kf_update_cov(
                            detected.covs, h_mat, this_observation_noise,
                            INPLACE=False)
                    else:
                        this_detected_covs = _this_detected_covs_.copy()
                    #new_x = copy.deepcopy(x)
                    # Apply the Kalman update to get the new state - 
                    # update in-place and return the residuals
                    upd_states, residuals = kf_update_x(
                        detected.states, pred_z, _observation_, 
                        kalman_info.kalman_gain, INPLACE=False)
                    # Calculate the weight of the Gaussians for this observation
                    # Calculate term in the exponent
                    #x_pdf = np.exp(-0.5*np.power(
                    #    blas.dgemv(kalman_info.inv_sqrt_S, 
                    #               residuals, TRANSPOSE_A=True), 2).sum(axis=1))/ \
                    #    np.sqrt(kalman_info.det_S*(2*np.pi)**z_dim)
                    x_pdf = misctools.approximate_mvnpdf(_observation_, pred_z, kalman_info.S)
                    upd_weights = detected.weights*x_pdf
                    #print "Clutter intensity = ", clutter_intensity[obs_count]
                    #print "Sum of weights    = ", upd_weights.sum()
                    #code.interact(local=locals())
                    # Normalise the weights
                    normalisation_factor = ( clutter_intensity[obs_count] + 
                                             self.vars.birth_intensity +
                                             upd_weights.sum() )
                    upd_weights /= normalisation_factor
                    #print "Obs Index: ", str(obs_count+1)
                    #print upd_weights.sum()
                    # SLAM, step 2:
                    self.slam_info.sum__clutter_with_pd_updwt[obs_count] = \
                        normalisation_factor
                    
                    # Create new state with new_x and P to add to _states_
                    updated.weights += [upd_weights.copy()]
                    updated.states += [upd_states.copy()]
                    updated.covs += [this_detected_covs]
                    #print upd_weights.sum()
            else:
                self.slam_info.sum__clutter_with_pd_updwt = np.ones(num_observations)
            
            self.weights = np.concatenate(updated.weights)
            self.states = np.concatenate(updated.states)
            self.covs = np.concatenate(updated.covs)
            
            # SLAM, finalise:
            self.slam_info.log_likelihood = (
                np.log(self.slam_info.exp_sum__pd_predwt) +
                np.log(self.slam_info.sum__clutter_with_pd_updwt).sum())
            self.slam_info.likelihood = (self.slam_info.exp_sum__pd_predwt * 
                                    self.slam_info.sum__clutter_with_pd_updwt.prod())
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            print "error in update"
            exc_info = sys.exc_info()
            print "GMPHD:UPDATE():\n", traceback.print_tb(exc_info[2])
            raise
        return self.slam_info
    
    def estimate(self, USE_INTENSITY=True):
        """phd.estimate -> (weights, states, covariances)
        Estimate the number of landmarks in the map and return as a tuple of
        the weights, states and covariances of the landmarks.
        """
        if not self.flags.ESTIMATE_IS_VALID:
            try:
                weights = self.weights.copy()
                states = self.states.copy()
                covs = self.covs.copy()
            except:
                exc_info = sys.exc_info()
                print "GMPHD:ESTIMATE():\n", traceback.print_tb(exc_info[2])
                raise
            if USE_INTENSITY:
                intensity = weights.sum()
                num_targets = min((intensity, weights.shape[0]))
                num_targets = int(round(num_targets))
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
                self._estimate_ = SAMPLE(est_weights, est_states, est_covs)
            else:
                self._estimate_ = SAMPLE(np.zeros(0), 
                                         np.zeros((0, 3)), np.zeros((0, 3, 3)))
            self.flags.ESTIMATE_IS_VALID = True
        return self._estimate_
        #return SAMPLE(self._estimate_.weight.copy(), 
        #              self._estimate_.state.copy(), 
        #              self._estimate_.covariance.copy())
    
    def prune(self, override_prune_threshold=-1, 
              override_max_num_components=-1):
        """phd.prune()
        Remove landmarks in the map with low weights.
        """
        try:
            # Get the threshold for weight based pruning
            prune_threshold = self.vars.prune_threshold
            if override_prune_threshold >= 0:
                prune_threshold = override_prune_threshold
            max_num_components = self.vars.max_num_components
            if override_max_num_components >= 0:
                max_num_components = override_max_num_components
            # Check if we need to prune
            if ((not self.weights.shape[0]) or 
                ((prune_threshold <= 0) and 
                 (self.weights.shape[0] < max_num_components))):
                return
            
            self.flags.ESTIMATE_IS_VALID = False
            if prune_threshold > 0:
                valid_idx = self.weights >= prune_threshold
                self.weights = self.weights[valid_idx]
                self.states = self.states[valid_idx]
                self.covs = self.covs[valid_idx]
            
            # Prune to the maximum number of components
            if ((max_num_components > 0) and 
                (self.weights.shape[0] > max_num_components)):
                print "Hit maximum number of components, pruning..."
                # Sort in ascending order and 
                # select the last max_num_components states
                idx = self.weights.argsort()[-max_num_components:]
                self.weights = self.weights[idx]
                self.states = self.states[idx]
                self.covs = self.covs[idx]
            assert (self.weights.shape[0] == 
                    self.states.shape[0] == 
                    self.covs.shape[0]), "Lost states!!"
        except:
            exc_info = sys.exc_info()
            print "GMPHD:PRUNE():\n", traceback.print_tb(exc_info[2])
            raise
    
    def merge_fov(self, detection_threshold=0.3):
        """phd.merge_fov(detection_threshold=0.5)
        Merge Gaussian components which are in the field of view or which
        satisfy a probability of detection given by detection_threshold.
        Set detection_threshold to 0 to merge landmarks everywhere in the map.
        """
        try:
            if (self.vars.merge_threshold < 0) or (self.weights.shape[0] < 2):
                return
            # Get objects in the field of view
            if detection_threshold <= 0:
                undetected_idx = np.empty(0, dtype=np.int)
                detected_idx = np.arange(self.states.shape[0])
            else:
                camera = self.sensors.camera
                # Convert states to camera coordinate system
                try:
                    # Stereo camera - get relative position for left and right
                    detected_idx = camera.pdf_detection(self.states)
                except:
                    exc_info = sys.exc_info()
                    print "Error merging states"
                    print "GMPHD:MERGE_FOV():\n", traceback.print_tb(exc_info[2])
                    raise
                undetected_idx = np.where(detected_idx < detection_threshold)[0]
                detected_idx = np.where(detected_idx >= detection_threshold)[0]
            #undetected_idx = misctools.gen_retain_idx(self.weights.shape[0], 
            #                                          detected_idx)
            
            if not detected_idx.shape[0]:
                return
            merged_wts = []
            merged_sts = []
            merged_cvs = []
            
            # Save samples which are not going to be merged
            unmerged_wts = self.weights[undetected_idx]
            unmerged_sts = self.states[undetected_idx]
            unmerged_cvs = self.covs[undetected_idx]
            # Remove unmerged samples from the state
            self.weights = self.weights[detected_idx]
            self.states = self.states[detected_idx]
            self.covs = self.covs[detected_idx]
            #num_remaining_components = self.weights.shape[0]
            
            merged_wts, merged_sts, merged_cvs = rfs_merge(self.weights, self.states, self.covs, self.vars.merge_threshold)
            #_mwt_ = np.asarray(_mwt_)
            #_mst_ = np.asarray(_mst_)
            #_mcv_ = np.asarray(_mcv_)
            
            """
            while num_remaining_components:
                max_wt_index = self.weights.argmax()
                max_wt_state = self.states[np.newaxis, max_wt_index]
                max_wt_cov = self.covs[np.newaxis, max_wt_index]
                mahalanobis_fn = misctools.approximate_mahalanobis
                mahalanobis_dist = mahalanobis_fn(max_wt_state, max_wt_cov, 
                                                  self.states)
                merge_list_indices = ( np.where(mahalanobis_dist <= 
                                                    self.vars.merge_threshold)[0] )
                retain_idx = ( np.where(mahalanobis_dist > 
                                                    self.vars.merge_threshold)[0] )
                new_wt, new_st, new_cv = misctools.merge_states(
                                                self.weights[merge_list_indices], 
                                                self.states[merge_list_indices],
                                                self.covs[merge_list_indices])
                merged_wts += [new_wt]
                merged_sts += [new_st]
                merged_cvs += [new_cv]
                # Remove merged states from the list
                #retain_idx = misctools.gen_retain_idx(self.weights.shape[0], 
                #                                      merge_list_indices)
                self.weights = self.weights[retain_idx]
                self.states = self.states[retain_idx]
                self.covs = self.covs[retain_idx]
                num_remaining_components = self.weights.shape[0]
            
            mwts = np.asarray(merged_wts)
            msts = np.asarray(merged_sts)
            mcvs = np.asarray(merged_cvs)
            if not np.allclose(_mwt_, mwts):
                print "Weights not equal"
                code.interact(local=locals())
            
            if not np.allclose(_mst_, msts):
                print "States not equal"
                code.interact(local=locals())
            
            if not np.allclose(_mcv_, mcvs):
                print "Covs not equal"
                code.interact(local=locals())
            """
            self.flags.ESTIMATE_IS_VALID = False
            self.set_states(np.hstack((unmerged_wts, np.asarray(merged_wts))), 
                            np.vstack((unmerged_sts, np.asarray(merged_sts))), 
                            np.vstack((unmerged_cvs, np.asarray(merged_cvs))))
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            exc_info = sys.exc_info()
            print "GMPHD:MERGE_FOV():\n", traceback.print_tb(exc_info[2])
            raise
    
    def compute_distances(self):
        """phd.compute_distances() -> distance_matrix
        Calculate the Mahalanobis distance between the landmarks.
        """
        distance_matrix = np.array(
            [misctools.mahalanobis(_x_, _P_[np.newaxis], self.states) 
            for (_x_, _P_) in zip(self.states, self.covs)])
        return distance_matrix
    
    def append(self, weights, states, covs):
        """phd.append(weights, states, covs)
        Add new weights, states and covariances to the Gaussian mixture.
        """
        try:
            self.flags.ESTIMATE_IS_VALID = False
            self.weights = np.hstack((self.weights, weights))
            if DEBUG:
                assert states.ndim == 2, "states must be a nxm ndarray"
                assert covs.ndim == 3, "covs must be a nxmxm ndarray"
            self.states = np.vstack((self.states, states))
            self.covs = np.vstack((self.covs, covs))
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            exc_info = sys.exc_info()
            print "GMPHD:APPEND():\n", traceback.print_tb(exc_info[2])
            raise
    
    #####################################
    ## Default iteration of PHD filter ##
    #####################################
    def iterate(self, observations, obs_noise):
        """phd.iterate(observations, obs_noise)
        Perform a single iteration of the filter:
            predict()
            update()
            merge()
            prune()
        """
        # Predict
        self.predict()
        # Update
        slam_info = self.update(observations, obs_noise)
        # Remove states with very low weights, but keep all the rest
        #self.prune(override_prune_threshold=self.vars.prune_threshold/10, 
        #           override_max_num_components=np.inf)
        # Add birth terms
        if observations.shape[0]:
            self.birth(observations, obs_noise, APPEND=True)
        # Prune extremely small components
        #self.prune(override_prune_threshold=self.vars.prune_threshold/10,
        #           override_max_num_components=np.inf)
        # Perform normal prune
        self.prune(override_max_num_components=np.inf)
        if self.weights.shape[0] > self.vars.prune_threshold:
            self.merge_fov(-1)
            self.prune()
        else:
            # Merge states in the field of view - too expensive to merge whole
            # state space
            self.merge_fov()
        # Generate vehicle state and map estimates
        self.estimate()
        # Perform normal prune
        #self.prune()
        return slam_info
    
    def intensity(self):
        """phd.intensity() -> intensity
        Compute the intensity of the PHD. This is a measure of the number of
        targets being tracked by the filter.
        """
        return self.weights.sum()
    
    def camera_birth(self, parent_ned, parent_rpy, features_rel, 
                     features_cv=None):
        """phd.camera_birth(parent_ned, parent_rpy, features_rel, 
            features_cv=None) -> birth_weights, birth_states, birth_covariances
        Create birth components using features visible from the camera.
        parent_ned - numpy array of parent position (north, east, down)
        parent_rpy - numpy array of parent orientation (roll, pitch, yaw)
        features_rel - Nx3 numpy array of feature positions relative to parent
        features_cv - Nx3x3 numpy array of covariance of the features
        """
        camera = self.sensors.camera
        birth_wt = np.empty(0)
        birth_st = np.empty((0, 3))
        birth_cv = np.empty((0, 3, 3))
        if features_rel.shape[0]:
            # Select features which are not on the edge of the FoV
            #visible_idx = camera.is_visible_relative2sensor(features_rel, 
            #    None, margin=0)
            birth_features = features_rel #[visible_idx]
            features_cv = features_cv #[visible_idx]
            if birth_features.shape[0]:
                birth_wt = self.vars.birth_intensity*np.ones(birth_features.shape[0])
                try:
                    birth_st = camera.to_world_coords(birth_features)
                except:
                    print "tf conversion to world coords failed!"
                    exc_info = sys.exc_info()
                    print "GMPHD:CAMERA_BIRTH():\n", traceback.print_tb(exc_info[2])
                    raise
                else:
                    if features_cv is None:
                        features_cv = np.repeat([np.eye(3)], birth_features.shape[0], 0)
                    else:
                        features_cv = features_cv.copy()
                    birth_cv = features_cv
        return (birth_wt, birth_st, birth_cv)
        
    def camera_pd(self, parent_ned, parent_rpy, features_abs):
        """phd.camera_pd(parent_ned, parent_rpy, features_abs) -> pd
        Returns the probability of detection (pd) of all landmarks in the map.
        parent_ned - numpy array of parent position (north, east, down)
        parent_rpy - numpy array of parent orientation (roll, pitch, yaw)
        features_abs - Nx3 numpy array of absolute position of features
        """
        camera = self.sensors.camera
        try:
            pdf_detection = camera.pdf_detection(features_abs)
        except:
            print "Error calling camera pdf_detection()"
            exc_info = sys.exc_info()
            print "GMPHD:CAMERA_PD():\n", traceback.print_tb(exc_info[2])
            pdf_detection = np.zeros(features_abs.shape[0])
        return pdf_detection
    
    def camera_clutter(self, observations):
        """phd.camera_clutter(observations) -> clutter_intensity
        Returns the clutter intensity evaluated for the observations
        observations - Nx3 numpy array indicating observations from landmarks
        """
        return self.sensors.camera.pdf_clutter(observations)
        
    def map_to_image_points(self):
        is_visible = self.sensors.camera.is_visible(self.states)
        is_visible[self.weights<0.4] = False
        if is_visible.shape[0]:
            camera_points = self.sensors.camera.from_world_coords(self.states[is_visible])
            img_points = self.sensors.camera.project3dToPixel(camera_points[0])
        else:
            img_points = (np.empty(0), np.empty(0))
        print "Generated image points"
        return img_points
    

###############################################################################
###############################################################################

def _update_features_parallel_(phdslam_object, idx, features_pos, features_noise):
    return phdslam_object.vehicle.maps[idx].iterate(features_pos, features_noise)
    

class PHDSLAM(object):
    def __init__(self, nparticles=5, ndims=6, resample_threshold=-1):
        """PHDSLAM() -> phdslam
        Creates an object that performs PHD SLAM in addition to updates from 
        odometry.
        This object is specific to the Girona500.
        """
        self.map_instance = GMPHD
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = True
        
        self.vars = STRUCT()
        self.vars.nparticles = nparticles#2*self.vars.ndims + 1
        self.vars.ndims = ndims
        self.vars.resample_threshold = resample_threshold
        
        self.vars.F = np.array(np.eye(self.vars.ndims)[np.newaxis], order='C')
        self.vars.Q = None
        self.vars.gpsH = None
        self.vars.dvlH = None
        self.vars.gpsR = None
        self.vars.dvl_w_R = None
        self.vars.dvl_b_R = None
        
        self.vehicle = STRUCT()
        self.vehicle.weights = 1.0/self.vars.nparticles * \
            np.ones(self.vars.nparticles)
        self.vehicle.states = np.zeros((self.vars.nparticles, self.vars.ndims))
        self.vehicle.covs = np.zeros((self.vars.nparticles, 
                                      self.vars.ndims, self.vars.ndims))
        # Store process noise for use with optimal sampling ratio
        self.vehicle.Q = None
        self.vehicle.maps = [self.map_instance() 
            for i in range(self.vars.nparticles)]
        
        if nparticles == 1:
            self._filter_update_ = self._kf_update_
        else:
            #self._filter_update_ = self._pf_update_
            self._filter_update_ = self._opt_pf_update_
        self.last_time = STRUCT()
        self.last_time.predict = 0
        #self.last_time.gps = 0
        #self.last_time.dvl = 0
        #self.last_time.svs = 0
        #self.last_time.map = 0
        
        self._estimate_ = STRUCT()
        self._estimate_.vehicle = STRUCT()
        self._estimate_.vehicle.ned = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.vehicle.vel_xyz = SAMPLE(1, np.zeros(3), 
                                                 np.zeros((3, 3)))
        self._estimate_.vehicle.rpy = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.map = SAMPLE(np.zeros(0), 
                                     np.zeros((0, 3)), np.zeros((0, 3, 3)))
    
    def set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R):
        """set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R)
        where Q is the model covariance, H is the observation matrix and R
        is the noise covariance matrix.
        dvl_b_R is the covariance for bottom lock and dvl_w_R is the covariance
        for water lock
        """
        self.vars.Q = np.asarray(Q, order='C')
        self.vars.gpsH = np.asarray(gpsH, order='C')
        self.vars.dvlH = np.asarray(dvlH, order='C')
        self.vars.gpsR = np.asarray(gpsR, order='C')
        self.vars.dvl_b_R = np.asarray(dvl_b_R, order='C')
        self.vars.dvl_w_R = np.asarray(dvl_w_R, order='C')
        
    def set_states(self, weights=None, states=None):
        """set_states(self, weights=None, states=None)
        Set new weights and states
        """
        if (weights is None) and (states is None):
            return
        new_weights = self.vehicle.weights if weights is None else weights
        new_states = self.vehicle.states if states is None else states
        self.vehicle.states[:, range(new_states.shape[1])] = new_states
        self.vehicle.weights = new_weights
    
    def get_position(self):
        return self.vehicle.states[:, :3]
    
    def reset_states(self):
        """reset_states(self)
        Reset states to zero
        """
        self.vehicle.states[:] = 0
        self.vehicle.weights = 1.0/float(self.vars.nparticles)* \
            np.ones(self.vars.nparticles)
    
    def trans_matrices(self, ctrl_input, delta_t, scalefac=1.):
        """trans_matrices(self, ctrl_input, delta_t)
        -> transition_matrix, scaled_process_noise
        Generate the transition and process noise matrices given the control
        input (roll, pitch, yaw) and the delta time
        """
        # Get process noise
        trans_mat = self.vars.F
        process_noise = self.vars.Q*scalefac
        
        # Multiply the rotation with delta_t to get F (or A)
        rot_mat = delta_t * rotation_matrix(ctrl_input)
        trans_mat[0, 0:3, 3:] = rot_mat
        
        # Multiply by delta_t/2 and delta_t to get W
        scale_matrix = np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))
        sc_process_noise = np.dot(scale_matrix, 
            np.dot(process_noise, scale_matrix.T)).squeeze() #\
            #+ delta_t/10*np.eye(6)
        #extra_scale_matrix = np.vstack((rot_mat*(delta_t/2, delta_t*np.eye(3)))
        extra_noise = np.min(np.diag(sc_process_noise))/1e6
        sc_process_noise += extra_noise*np.eye(sc_process_noise.shape[0])
        return trans_mat, sc_process_noise
    
    def _copy_state_to_map_(self, parent_ned=None, ctrl_input=None, rot_matrix=None):
        for i in range(self.vars.nparticles):
            if not parent_ned is None:
                self.vehicle.maps[i].parent_ned = parent_ned[i]
            if not ctrl_input is None:
                self.vehicle.maps[i].parent_rpy = ctrl_input
            if not rot_matrix is None:
                self.vehicle.maps[i].vars.H = rot_matrix
    
    def predict(self, ctrl_input, predict_to_time):
        """predict(self, ctrl_input, predict_to_time)
        Predict the state to the specified time given the control input
        (roll, pitch, yaw)
        """
        if self.last_time.predict == 0:
            self.last_time.predict = predict_to_time
            return
        delta_t = predict_to_time - self.last_time.predict
        self.last_time.predict = predict_to_time
        if delta_t < 0:
            return
        
        # Predict states
        trans_mat, sc_process_noise = self.trans_matrices(ctrl_input, delta_t)
        
        #pred_states = blas.dgemv(trans_mat, self.vehicle.states)
        pred_states = np.array( 
            np.dot(trans_mat[0], self.vehicle.states.T).T, order='C')
        self.vehicle.states = pred_states
        
        # Only add process noise if using a particle filter (more than 1 state)
        self.vehicle.ctrl_input = ctrl_input
        self.vehicle.delta_t = delta_t
        _tm_, self.vehicle.Qprior = self.trans_matrices(ctrl_input, delta_t, scalefac=10000**2)
        _tm_, self.vehicle.Qsample = self.trans_matrices(ctrl_input, delta_t, scalefac=100**2)
        if ((pred_states.shape[0] > 1) and 
            (self._filter_update_ == self._pf_update_)):
            proc_noise = np.random.multivariate_normal(np.zeros(6), sc_process_noise, pred_states.shape[0])
            self.vehicle.states += proc_noise
        
        # Predict covariance
        self.vehicle.covs = kf_predict_cov(self.vehicle.covs, trans_mat, 
                                          sc_process_noise)
        
        # Copy the particle state to the PHD parent state
        parent_ned = np.array(pred_states[:, 0:3])
        #Calculate the rotation matrix to store for the map update
        rot_mat = rotation_matrix(ctrl_input)
        # Copy the predicted states to the "parent state" attribute and 
        # perform a prediction for the map
        self._copy_state_to_map_(parent_ned, ctrl_input, rot_mat)
        #for i in range(self.vars.nparticles):
        #    self.vehicle.maps[i].parent_ned = parent_ned[i]
        #    self.vehicle.maps[i].parent_rpy = ctrl_input
        #    self.vehicle.maps[i].vars.H = rot_mat
        #    # self.vehicle.maps.predict()  # Not needed
    
    def _kf_update_(self, weights, states, covs, h_mat, r_mat, z):
        """_kf_update_(self, weights, states, covs, h_mat, r_mat, z)
        Kalman filter update
        """
        # predicted observations
        #pred_z = blas.dgemv(h_mat, states)
        pred_z = np.array(np.dot(h_mat, states.T).T, order='C')
        # covariance is the same for all states, do the update for one matrix
        
        upd_cov0, kalman_info = np_kf_update_cov(covs[0], h_mat, r_mat, False)
        
        upd_covs = np.repeat(upd_cov0[np.newaxis], covs.shape[0], axis=0)
        # Update the states
        upd_states, residuals = kf_update_x(states, pred_z, z, 
            np.array([kalman_info.kalman_gain], order='C'))
        if not upd_states.flags.c_contiguous:
            upd_states = np.array(upd_states, order='C')
        # Evaluate the new weight
        #x_pdf = np.exp(-0.5*np.power(
        #    blas.dgemv(np.array([kalman_info.inv_sqrt_S]), residuals), 2).sum(axis=1))/ \
        #    np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        x_pdf = np.exp(-0.5*np.power(
            np.dot(kalman_info.inv_sqrt_S, residuals.T).T, 2).sum(axis=1))/ \
            np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        upd_weights = weights * x_pdf
        upd_weights /= upd_weights.sum()
        return upd_weights, upd_states, upd_covs
    
    def _opt_pf_update_(self, weights, states, covs, h_mat, r_mat, z):
        """_opt_pf_update_(self, weights, states, covs, h_mat, r_mat, z)
        Particle filter update with optimal proposal
        """
        # Optimal proposal is the mean and covariance of the KF applied to the state
        # updated covariance - same for all states
        try:
            upd_cov0, kalman_info = np_kf_update_cov(covs[0], h_mat, r_mat*16, False)
            upd_covs = np.repeat(upd_cov0[np.newaxis], covs.shape[0], axis=0)
            # Update the states using the KF
            kf_pred_z = np.array(np.dot(h_mat, states.T).T, order='C')
            kf_mean_states, residuals = kf_update_x(states, kf_pred_z, z, 
                np.array([kalman_info.kalman_gain], order='C'))
            # Draw the updated states from the KF distribution
            # process noise
            #code.interact(local=locals())
            # Importance distribution covariance
            #imp_q, _ki_ = np_kf_update_cov(self.vehicle.Q, h_mat, r_mat, False)
            #imp_q = np.asarray(imp_q[np.newaxis], order='C')
            imp_q = np.asarray(self.vehicle.Qsample[np.newaxis], order='C')
            prior_q = np.asarray(self.vehicle.Qprior[np.newaxis], order='C')
            # Draw samples from the distribution
            q_noise = np.random.multivariate_normal(
                np.zeros(kf_mean_states.shape[1]), imp_q[0], (kf_mean_states.shape[0]))
                #np.zeros(kf_mean_states.shape[1]), upd_cov0, (kf_mean_states.shape[0]))
            # Generate the predicted states
            upd_states = kf_mean_states + q_noise
            # Evaluate likelihood
            pred_z = np.array(np.dot(h_mat, upd_states.T).T, order='C')
            x_pdf = misctools.mvnpdf(pred_z, z[np.newaxis], np.asarray(r_mat[np.newaxis], order='C'), LOG=True)
            
            # Evaluate the importance sampling weight p()/q()
            # Difference between the predicted state (no noise) with the sampled state
            #p_noise = upd_states - states
            p_prior = misctools.mvnpdf(
                upd_states, states, prior_q, LOG=True)
            p_sampling = misctools.mvnpdf(
                #q_noise, np.zeros((1, q_noise.shape[1])), Q, LOG=True)
                q_noise, np.zeros((1, q_noise.shape[1])),
                imp_q, LOG=True)
            # Updated importance weight is x_pdf*p/q
            log_update = p_prior - p_sampling + x_pdf
            # Normalise by the largest number to avoid underflows
            log_update -= np.max(log_update)
            upd_weights = np.exp(log_update)
            # Multiply updated weights by previous
            upd_weights *= weights
            #upd_weights = weights*np.exp(x_pdf)
            # Normalise to get a pdf
            upd_weights /= upd_weights.sum()
        except:
            exc_info = sys.exc_info()
            print "GMPHD:_OPT_PF_UPDATE_():\n", traceback.print_tb(exc_info[2])
            print "Error in update"
            raise
        return upd_weights, upd_states, upd_covs
        
    def _pf_update_(self, weights, states, covs, h_mat, r_mat, z):
        """_pf_update_(self, weights, states, covs, h_mat, r_mat, z)
        Particle filter update
        """
        pred_z = np.array(np.dot(h_mat, states.T).T, order='C')
        x_pdf = misctools.mvnpdf(pred_z, z[np.newaxis], np.asarray(r_mat[np.newaxis], order='C'), LOG=True)
        upd_states = states.copy()
        upd_cov0, kalman_info = np_kf_update_cov(covs[0], h_mat, r_mat, False)
        upd_covs = np.repeat(upd_cov0[np.newaxis], covs.shape[0], axis=0)
        x_pdf -= x_pdf.max()
        upd_weights = weights * np.exp(x_pdf)
        upd_weights /= upd_weights.sum()
        return upd_weights, upd_states, upd_covs
        
    def update_gps(self, gps):
        """update_gps(self, gps)
        Update the state using the gps measurement
        """
        self.flags.ESTIMATE_IS_VALID = False
        h_mat = self.vars.gpsH #np.array([self.vars.gpsH])
        r_mat = self.vars.gpsR #np.array([self.vars.gpsR])
        upd_weights, upd_states, upd_covs = \
                self._filter_update_(self.vehicle.weights, self.vehicle.states, 
                                 self.vehicle.covs, h_mat, r_mat, gps)
        self.vehicle.weights = upd_weights
        self.vehicle.states = upd_states
        self.vehicle.covs = upd_covs
        # Copy the particle state to the PHD parent state
        parent_ned = np.array(self.vehicle.states[:, 0:3])
        self._copy_state_to_map_(parent_ned)
    
    def update_dvl(self, dvl, mode):
        """update_dvl(self, dvl, mode)
        Update the vehicle state using the dvl measurement.
        mode = 'b' or 'w' for bottom or water lock
        """
        self.flags.ESTIMATE_IS_VALID = False
        assert mode in ['b', 'w'], "Specify (b)ottom or (w)ater for dvl update"
        if mode == 'b':
            r_mat = self.vars.dvl_b_R #np.array([self.vars.dvl_b_R])
        else:
            r_mat = self.vars.dvl_w_R #np.array([self.vars.dvl_w_R])
        h_mat = self.vars.dvlH #np.array([self.vars.dvlH])
        upd_weights, upd_states, upd_covs = \
                self._filter_update_(self.vehicle.weights, self.vehicle.states, 
                                 self.vehicle.covs, h_mat, r_mat, dvl)
        self.vehicle.weights = upd_weights
        self.vehicle.states = upd_states
        self.vehicle.covs = upd_covs
        # Copy the particle state to the PHD parent state
        parent_ned = np.array(self.vehicle.states[:, 0:3])
        self._copy_state_to_map_(parent_ned)
    
    def update_svs(self, svs):
        """update_svs(self, svs)
        Update the state using the depth reading
        """
        self.flags.ESTIMATE_IS_VALID = False
        self.vehicle.states[:, 2] = svs
        # Copy the particle state to the PHD parent state
        parent_ned = np.array(self.vehicle.states[:, 0:3])
        self._copy_state_to_map_(parent_ned)
    
    def update_features(self, features):
        """update_features(self, features)
        Update the map using the features. Features are specified as a Nx6
        numpy array where the first 3 columns are x, y, z positions and the
        last 3 columns are the diagonal of the covariance matrix for that
        measurement
        """
        self.flags.ESTIMATE_IS_VALID = False
        if features.shape[0]:
            features_pos = features[:, 0:3].copy()
            features_noise = np.array([np.diag(features[i, 3:6]) 
                for i in range(features.shape[0])])
        else:
            features_pos = np.empty((0, 3))
            features_noise = np.empty((0, 3, 3))
        slam_info = [self.vehicle.maps[i].iterate(features_pos, features_noise) 
            for i in range(self.vars.nparticles)]
        #slam_info = Parallel(n_jobs=2)(delayed(_update_features_parallel_)(self, idx, features_pos, features_noise) for idx in range(self.vars.nparticles))
        print "map size: ", self.vehicle.maps[0].weights.shape[0]
        # Overflow errors
        #slam_weight_update = np.array([slam_info[i].likelihood
        #    for i in range(self.vars.nparticles)])
        #self.vehicle.weights *= slam_weight_update/slam_weight_update.sum()
        #self.vehicle.weights /= self.vehicle.weights.sum()
        # Use log to avoid overflows
        slam_weight_log_update = np.array([slam_info[i].log_likelihood
            for i in range(self.vars.nparticles)])
        log_weights = np.log(self.vehicle.weights)
        slam_weight_log_update -= slam_weight_log_update.max()
        slam_weight_log_update_sum = np.log(np.sum(np.exp(slam_weight_log_update)))
        log_weights += slam_weight_log_update - slam_weight_log_update_sum
        self.vehicle.weights = np.exp(log_weights)
        self.vehicle.weights /= self.vehicle.weights.sum()
        # Create birth terms
        #if features.shape[0]:
        #    b_wt, b_st, b_cv = self.vehicle.maps[0].birth(features_pos, 
        #        features_noise, APPEND=False)
        #    nothing = [_map_.append(b_wt, b_st, b_cv) for _map_ in self.vehicle.maps]
        #print "post update weights = ", self.vehicle.weights
    
    def compress_maps(self, *args, **kwargs):
        """compress_maps(self, *args, **kwargs)
        Merge all possible landmarks in the maps
        """
        [_map_.merge_fov(-1) for _map_ in self.vehicle.maps]
    
    def estimate(self):
        """estimate(self) -> estimate
        Generate the state and map estimates
        """
        if not self.flags.ESTIMATE_IS_VALID:
            if self.vars.nparticles == 1:
                self._estimate_.vehicle.ned = SAMPLE(1, self.vehicle.states[0, 0: 3],
                                                     self.vehicle.covs[0, 0:3, 0:3])
                max_weight_idx = 0
            else:
                state, cov = misctools.sample_mn_cv(self.vehicle.states, 
                                                    self.vehicle.weights)
                self._estimate_.vehicle.ned = SAMPLE(1, state[0: 3], cov[0:3, 0:3])
                max_weight_idx = np.argmax(self.vehicle.weights)
            self._estimate_.map = self.vehicle.maps[max_weight_idx].estimate()
            """
            iter_range = range(self.vars.nparticles)
            map_estimates = [self.vehicle.maps[i].estimate() for i in iter_range]
            est_gmphd = GMPHD()
            est_gmphd.vars.merge_threshold = 1
            for i in iter_range:
                est_gmphd.append(map_estimates[i].weight*self.vehicle.weights[i], 
                                 map_estimates[i].state, 
                                 map_estimates[i].covariance)
            est_gmphd.merge()
            self._estimate_.map = est_gmphd.estimate()
            """
            self.flags.ESTIMATE_IS_VALID = True
        return self._estimate_
    
    def resample(self):
        """resample(self)
        Resample the particles according to self.vars.resample_threshold
        """
        # Effective number of particles
        nparticles = self.vars.nparticles
        eff_nparticles = 1/np.power(self.vehicle.weights, 2).sum()
        resample_threshold = eff_nparticles/nparticles
        # Check if we have particle depletion
        if (resample_threshold > self.vars.resample_threshold):
            return
        print "Resampling"
        if np.any(np.isnan(self.vehicle.weights)):
            print "\n\n***  NaNs in weights  ***\n\n"
        # Otherwise we need to resample
        # Find the mean and covariance of the particles
        #states_mn = np.mean(self.vehicle.states, axis=0)
        #states_cv = np.var(self.vehicle.states, axis=0)*9*np.eye(states_mn.shape[0])
        #gauss_likelihood = misctools.approximate_mvnpdf(self.vehicle.states,
        #    states_mn[np.newaxis], states_cv[np.newaxis])
        #gauss_likelihood /= gauss_likelihood.sum()
        #self.vehicle.weights *= gauss_likelihood
        resample_index = misctools.get_resample_index(self.vehicle.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        self.vehicle.weights = np.ones(nparticles, dtype=float)/nparticles
        self.vehicle.states = np.asanyarray(self.vehicle.states[resample_index], order='C')
        self.vehicle.covs = np.asanyarray(self.vehicle.covs[resample_index], order='C')
        
        # Only resample the maps if they are populated
        num_map_components = np.asarray([_map_.weights.shape[0] for _map_ in self.vehicle.maps])
        if np.any(num_map_components):
            resampled_maps = [self.vehicle.maps[i].copy() for i in resample_index]
            for count in range(nparticles):
                resampled_maps[count].sensors.camera.set_tf_frame(
                    *self.vehicle.maps[count].sensors.camera.get_tf_frame())
            self.vehicle.maps = resampled_maps
    
    #def resample_hack(self):
    #    """resample_hack(self)
    #    Reinitialise sigma states when number of effective particles is 1
    #    """
    #    # Effective number of particles
    #    eff_nparticles = 1/np.power(self.vehicle.weights, 2).sum()
    #    if eff_nparticles <= 1.5
        

class RBPHDSLAM(object):
    def __init__(self, nparticles=1000, ndims=None, resample_threshold=0.5):
        """RBPHDSLAM() -> phdslam
        Creates an object that performs RB PHD SLAM in addition to updates from 
        odometry.
        This object is specific to the Girona500.
        """
        self.map_instance = GMPHD
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = True
        
        self.vars = STRUCT()
        self.vars.nparticles = nparticles
        self.vars.resample_threshold = resample_threshold
        
        self.vars.F = None
        self.vars.Q = None
        self.vars.gpsH = None
        self.vars.dvlH = None
        self.vars.gpsR = None
        self.vars.dvl_w_R = None
        self.vars.dvl_b_R = None
        
        self.vehicle = STRUCT()
        # Kalman filter is used to estimate the velocity
        self.vehicle.velocity = None #np.zeros(3)
        self.vehicle.velocity_cov = None #np.zeros((3, 3))
        # Vehicle orientation (RPY)
        self.vehicle.orientation = np.zeros(3)
        # Particle representation for the vehicle position
        # Weights for the particles
        self.vehicle.weights = None
        # Particles representing the x,y,z position
        self.vehicle.position = None
        # Map of landmarks, one for each particle
        self.vehicle.maps = [self.map_instance() 
                             for i in range(self.vars.nparticles)]
        self.reset_states()
        
        self.last_time = STRUCT()
        self.last_time.predict = 0
        
        # Structure for storing estimated pose and map
        self._estimate_ = STRUCT()
        self._estimate_.vehicle = STRUCT()
        self._estimate_.vehicle.ned = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.vehicle.vel_xyz = SAMPLE(1, np.zeros(3), 
                                                 np.zeros((3, 3)))
        self._estimate_.vehicle.rpy = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.map = SAMPLE(np.zeros(0), 
                                     np.zeros((0, 3)), np.zeros((0, 3, 3)))
    
    def set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R):
        """set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R)
        where Q is the model covariance, H is the observation matrix and R
        is the noise covariance matrix.
        dvl_b_R is the covariance for bottom lock and dvl_w_R is the covariance
        for water lock
        """
        self.vars.Q = np.asarray(Q, order='C')
        self.vars.gpsH = np.asarray(gpsH, order='C')
        self.vars.dvlH = np.asarray(dvlH, order='C')
        self.vars.gpsR = np.asarray(gpsR, order='C')
        self.vars.dvl_b_R = np.asarray(dvl_b_R, order='C')
        self.vars.dvl_w_R = np.asarray(dvl_w_R, order='C')
    
    def set_states(self, weights=None, states=None):
        """set_states(self, weights=None, states=None)
        Set new weights and states
        """
        if (weights is None) and (states is None):
            return
        new_weights = self.vehicle.weights if weights is None else weights
        new_states = self.vehicle.states if states is None else states
        self.vehicle.weights[:] = new_weights
        self.vehicle.position[:, range(new_states.shape[1])] = new_states
    
    def get_position(self):
        return self.vehicle.position
    
    def reset_states(self):
        """reset_states(self)
        Reset states to zero
        """
        # Kalman filter is used to estimate the velocity
        self.vehicle.velocity = np.zeros(3)
        self.vehicle.velocity_cov = np.zeros((3, 3))
        
        # Particle representation for the vehicle position
        # Weights for the particles
        self.vehicle.weights = (
            1.0/self.vars.nparticles *np.ones(self.vars.nparticles))
        # Particles representing the x,y,z position
        self.vehicle.position = np.zeros((self.vars.nparticles, 3))
    
    def trans_matrices(self, ctrl_input, delta_t):
        """trans_matrices(self, ctrl_input, delta_t)
        -> ((kf_transition_matrix, kf_scaled_process_noise), 
            (pf_transition_matrix, pf_scaled_process_noise))
        Generate the transition and process noise matrices given the control
        input (roll, pitch, yaw) and the delta time
        """
        
        # Multiply the rotation with delta_t to get F (or A)
        rot_mat = delta_t * rotation_matrix(ctrl_input)
        # Transition matrix corresponding to x,y,z (particle filter)
        pf_trans_mat = np.hstack((np.eye(3), rot_mat))
        # Transition matrix corresponding to vx, vy, vz (Kalman filter)
        kf_trans_mat = np.eye(3)
        
        # Multiply by delta_t/2 and delta_t to get W
        scale_matrix = np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))
        sc_process_noise = np.dot(scale_matrix, 
            np.dot(self.vars.Q, scale_matrix.T)).squeeze()
        
        # Extract noise covariances for KF and PF from the joint matrix
        # Correlations between velocity and position are ignored
        pf_process_noise = np.asarray(
            sc_process_noise[np.ix_(range(3), range(3))], order='C')
        kf_process_noise = np.asarray(
            sc_process_noise[np.ix_(range(3, 6), range(3, 6))], order='C')
        
        # Combined transition matrix
        comb_trans_mat = np.vstack((pf_trans_mat, kf_trans_mat))
        return ((kf_trans_mat, kf_process_noise), 
                (pf_trans_mat, pf_process_noise),
                (comb_trans_mat, sc_process_noise))
    
    def _copy_state_to_map_(self, parent_ned=None, ctrl_input=None, rot_matrix=None):
        for i in range(self.vars.nparticles):
            if not parent_ned is None:
                self.vehicle.maps[i].parent_ned = parent_ned[i]
            if not ctrl_input is None:
                self.vehicle.maps[i].parent_rpy = ctrl_input
            if not rot_matrix is None:
                self.vehicle.maps[i].vars.H = rot_matrix
    
    def predict(self, ctrl_input, predict_to_time):
        """predict(self, ctrl_input, predict_to_time)
        Predict the state to the specified time given the control input
        (roll, pitch, yaw)
        """
        if self.last_time.predict == 0:
            self.last_time.predict = predict_to_time
            return
        delta_t = predict_to_time - self.last_time.predict
        self.last_time.predict = predict_to_time
        if delta_t < 0:
            return
        
        # Predict states - get transition matrices and process noise
        ((kf_trans_mat, kf_process_noise),
         (pf_trans_mat, pf_process_noise),
         (comb_trans_mat, sc_process_noise))= (
             self.trans_matrices(ctrl_input, delta_t))
        
        vehicle = self.vehicle
        # Kalman filter prediction
        # Predict the velocity - constant velocity model
        pred_velocity = np.dot(kf_trans_mat, vehicle.velocity)
        pred_velocity_cov = np_kf_predict_cov(
            vehicle.velocity_cov, kf_trans_mat, kf_process_noise)
        
        # Particle filter prediction
        # Draw samples from the velocity distribution
        velocity_samples = np.random.multivariate_normal(
            vehicle.velocity, pred_velocity_cov, self.vars.nparticles)
        rot_mat = pf_trans_mat[:, 3:]
        # Predict according to motion model
        pred_position = np.asarray(
            vehicle.position + np.dot(rot_mat, velocity_samples.T).T, order='C')
        # Generate noise
        position_noise = np.random.multivariate_normal(
            np.zeros(3), pf_process_noise, self.vars.nparticles)
        pred_position += position_noise
        
        # Save prediction
        vehicle.velocity = pred_velocity
        vehicle.velocity_cov = pred_velocity_cov
        vehicle.position = pred_position
        
        # Copy the predicted states to the "parent state" attribute and 
        # perform a prediction for the map
        self._copy_state_to_map_(pred_position, ctrl_input, rot_mat)
    
    def _filter_update_(self, h_mat, r_mat, z, INPLACE=True):
        # Kalman filter update for the velocity
        # predicted observations
        vehicle = self.vehicle
        # Predicted velocity measurements
        pred_z = np.array(np.dot(h_mat, vehicle.velocity.T).T, order='C')
        # Updated velocity covariance
        upd_velocity_cov, kalman_info = np_kf_update_cov(vehicle.velocity_cov,
                                                         h_mat, r_mat, INPLACE)
        
        # Update the states
        upd_velocity, residuals = np_kf_update_x(vehicle.velocity, pred_z, z,
            kalman_info.kalman_gain, INPLACE)
        
        if not INPLACE:
            return vehicle.weights, upd_velocity, upd_velocity_cov
        return None
    
    def update_gps(self, gps):
        pass
    
    def update_dvl(self, dvl, mode):
        """update_dvl(self, dvl, mode)
        Update the vehicle state using the dvl measurement.
        mode = 'b' or 'w' for bottom or water lock
        """
        self.flags.ESTIMATE_IS_VALID = False
        assert mode in ['b', 'w'], "Specify (b)ottom or (w)ater for dvl update"
        if mode == 'b':
            r_mat = self.vars.dvl_b_R #np.array([self.vars.dvl_b_R])
        else:
            r_mat = self.vars.dvl_w_R #np.array([self.vars.dvl_w_R])
        #h_mat = self.vars.dvlH #np.array([self.vars.dvlH])
        self._filter_update_(np.eye(3), r_mat, dvl, INPLACE=True)
        # Copy the particle state to the PHD parent state
        self._copy_state_to_map_(self.vehicle.position)
    
    def update_svs(self, svs):
        """update_svs(self, svs)
        Update the state using the depth reading
        """
        self.flags.ESTIMATE_IS_VALID = False
        self.vehicle.position[:, 2] = svs
        # Copy the particle state to the PHD parent state
        self._copy_state_to_map_(self.vehicle.position)
    
    def update_features(self, features):
        """update_features(self, features)
        Update the map using the features. Features are specified as a Nx6
        numpy array where the first 3 columns are x, y, z positions and the
        last 3 columns are the diagonal of the covariance matrix for that
        measurement
        """
        self.flags.ESTIMATE_IS_VALID = False
        if features.shape[0]:
            features_pos = features[:, 0:3].copy()
            features_noise = np.array([np.diag(features[i, 3:6]) 
                for i in range(features.shape[0])])
        else:
            features_pos = np.empty((0, 3))
            features_noise = np.empty((0, 3, 3))
        slam_info = [self.vehicle.maps[i].iterate(features_pos, features_noise) 
            for i in range(self.vars.nparticles)]
        #slam_info = Parallel(n_jobs=2)(delayed(_update_features_parallel_)(self, idx, features_pos, features_noise) for idx in range(self.vars.nparticles))
        print "map size: ", self.vehicle.maps[0].weights.shape[0]
        slam_weight_update = np.array([slam_info[i].likelihood
            for i in range(self.vars.nparticles)])
        self.vehicle.weights *= slam_weight_update/slam_weight_update.sum()
        self.vehicle.weights /= self.vehicle.weights.sum()
    
    def compress_maps(self, *args, **kwargs):
        """compress_maps(self, *args, **kwargs)
        Merge all possible landmarks in the maps
        """
        [_map_.merge_fov(-1) for _map_ in self.vehicle.maps]
    
    def estimate(self):
        """estimate(self) -> estimate
        Generate the state and map estimates
        """
        if not self.flags.ESTIMATE_IS_VALID:
            vehicle = self.vehicle
            position, cov = misctools.sample_mn_cv(vehicle.position, 
                                                   vehicle.weights)
            self._estimate_.vehicle.ned = SAMPLE(1, position, cov)
            max_weight_idx = vehicle.weights.argmax()
            self._estimate_.map = vehicle.maps[max_weight_idx].estimate()
            self.flags.ESTIMATE_IS_VALID = True
        return self._estimate_
    
    def resample(self):
        """resample(self)
        Resample the particles according to self.vars.resample_threshold
        """
        nparticles = self.vars.nparticles
        vehicle = self.vehicle
        # Effective number of particles
        eff_nparticles = 1/np.power(self.vehicle.weights, 2).sum()
        resample_threshold = eff_nparticles/nparticles
        # Check if we have particle depletion
        if (resample_threshold > self.vars.resample_threshold):
            return
        print "Resampling"
        if np.any(np.isnan(vehicle.weights)):
            print "\n\n***  NaNs in weights  ***\n\n"
        # Otherwise we need to resample
        resample_index = misctools.get_resample_index(vehicle.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        vehicle.weights = np.ones(nparticles, dtype=float)/nparticles
        vehicle.position = np.asarray(vehicle.position[resample_index], order='C')
        
        # Only resample the maps if they are populated
        num_map_components = np.asarray(
            [_map_.weights.shape[0] for _map_ in vehicle.maps])
        if np.any(num_map_components):
            resampled_maps = [self.vehicle.maps[i].copy() for i in resample_index]
            # Map copies will contain references to the previous frame id.
            # Reinitialise to unique ids
            for count in range(nparticles):
                resampled_maps[count].sensors.camera.set_tf_frame(
                    *self.vehicle.maps[count].sensors.camera.get_tf_frame())
            vehicle.maps = resampled_maps


class RBPHDSLAM2(object):
    def __init__(self, nparticles=1000, ndims=None, resample_threshold=0.5):
        """RBPHDSLAM() -> phdslam
        Creates an object that performs RB PHD SLAM in addition to updates from 
        odometry.
        This object is specific to the Girona500.
        """
        self.map_instance = GMPHD
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = True
        
        self.vars = STRUCT()
        self.vars.nparticles = nparticles
        self.vars._nparticles_ = nparticles
        self.vars.resample_threshold = resample_threshold
        
        self.vars.F = None
        self.vars.Q = None
        self.vars.gpsH = None
        self.vars.dvlH = None
        self.vars.gpsR = None
        self.vars.dvl_w_R = None
        self.vars.dvl_b_R = None
        if nparticles > 1:
            prediction_noise_scaling = rospy.get_param(
                "/phdslam/vars/prediction_noise_scaling0", 4)
        else:
            prediction_noise_scaling = 0
        self.vars.prediction_noise_scaling = prediction_noise_scaling
        
        self.vars.imm_ratio = rospy.get_param("/phdslam/vars/imm_ratio")
        self.vars.use_jms = rospy.get_param("/phdslam/vars/use_jms")
        
        self.vehicle = STRUCT()
        # Mode for model: constant position=0, constant velocity=1
        if self.vars.use_jms:
            self.vehicle.mode = np.random.randint(0, 2, self.vars.nparticles)
        else:
            self.vehicle.mode = np.ones(self.vars.nparticles)
            cp_nparticles = int(np.floor(self.vars.imm_ratio*self.vars.nparticles))
            self.vehicle.mode[:cp_nparticles] = 0
        
        # Kalman filter is used to estimate the velocity
        self.vehicle.kf_state = None #np.zeros(3)
        self.vehicle.kf_state_cov = None #np.zeros((3, 3))
        # Vehicle orientation (RPY)
        self.vehicle.orientation = np.zeros(3)
        # Particle representation for the vehicle position
        # Weights for the particles
        self.vehicle.weights = None
        # Particles representing the x,y,z position
        self.vehicle.position = None
        # Map of landmarks, one for each particle
        self.vehicle.maps = [self.map_instance() 
                             for i in range(self.vars.nparticles)]
        self.reset_states()
        
        self.last_time = STRUCT()
        self.last_time.predict = 0
        
        # Structure for storing estimated pose and map
        self._estimate_ = STRUCT()
        self._estimate_.vehicle = STRUCT()
        self._estimate_.vehicle.ned = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.vehicle.vel_xyz = SAMPLE(1, np.zeros(3), 
                                                 np.zeros((3, 3)))
        self._estimate_.vehicle.rpy = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.map = SAMPLE(np.zeros(0), 
                                     np.zeros((0, 3)), np.zeros((0, 3, 3)))
    
    def set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R):
        """set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R)
        where Q is the model covariance, H is the observation matrix and R
        is the noise covariance matrix.
        dvl_b_R is the covariance for bottom lock and dvl_w_R is the covariance
        for water lock
        """
        self.vars.Q = np.asarray(Q, order='C')
        self.vars.gpsH = np.asarray(gpsH, order='C')
        self.vars.dvlH = np.asarray(dvlH, order='C')
        self.vars.gpsR = np.asarray(gpsR, order='C')
        self.vars.dvl_b_R = np.asarray(dvl_b_R, order='C')
        self.vars.dvl_w_R = np.asarray(dvl_w_R, order='C')
    
    def set_states(self, weights=None, states=None):
        """set_states(self, weights=None, states=None)
        Set new weights and states
        """
        if (weights is None) and (states is None):
            return
        new_weights = self.vehicle.weights if weights is None else weights
        new_states = self.vehicle.states if states is None else states
        self.vehicle.weights[:] = new_weights
        self.vehicle.position[:, range(new_states.shape[1])] = new_states
    
    def get_position(self):
        return self.vehicle.position
    
    def reset_states(self):
        """reset_states(self)
        Reset states to zero
        """
        # Kalman filter is used to estimate the velocity
        self.vehicle.kf_state = np.zeros(6)
        self.vehicle.kf_state_cov = np.zeros((6, 6))
        
        # Particle representation for the vehicle position
        # Weights for the particles
        self.vehicle.weights = (
            1.0/self.vars.nparticles *np.ones(self.vars.nparticles))
        # Particles representing the x,y,z position
        self.vehicle.position = np.zeros((self.vars.nparticles, 3))
    
    def trans_matrices(self, ctrl_input, delta_t):
        """trans_matrices(self, ctrl_input, delta_t)
        -> ((kf_transition_matrix, kf_scaled_process_noise), 
            (pf_transition_matrix, pf_scaled_process_noise))
        Generate the transition and process noise matrices given the control
        input (roll, pitch, yaw) and the delta time
        """
        
        # Multiply the rotation with delta_t to get F (or A)
        rot_mat = delta_t * rotation_matrix(ctrl_input)
        # Transition matrix corresponding to x,y,z (particle filter)
        pf_trans_mat = np.hstack((np.eye(3), rot_mat))
        # Transition matrix corresponding to vx, vy, vz (Kalman filter)
        kf_trans_mat = np.eye(3)
        
        # Multiply by delta_t/2 and delta_t to get W
        scale_matrix = np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))
        sc_process_noise = np.dot(scale_matrix, 
            np.dot(self.vars.Q, scale_matrix.T)).squeeze()
        
        # Extract noise covariances for KF and PF from the joint matrix
        # Correlations between velocity and position are ignored
        pf_process_noise = np.asarray(
            sc_process_noise[np.ix_(range(3), range(3))], order='C')
        #kf_process_noise = np.asarray(
        #    sc_process_noise[np.ix_(range(3, 6), range(3, 6))], order='C')
        
        # Combined transition matrix
        comb_trans_mat = np.vstack((pf_trans_mat,
                                    np.hstack((np.zeros((3, 3)), kf_trans_mat))))
        return ((comb_trans_mat, sc_process_noise), 
                (pf_trans_mat, pf_process_noise))
    
    def _copy_state_to_map_(self, parent_ned=None, ctrl_input=None, rot_matrix=None):
        for i in range(self.vars.nparticles):
            if not parent_ned is None:
                self.vehicle.maps[i].parent_ned = parent_ned[i]
            if not ctrl_input is None:
                self.vehicle.maps[i].parent_rpy = ctrl_input
            if not rot_matrix is None:
                self.vehicle.maps[i].vars.H = rot_matrix
    
    def predict(self, ctrl_input, predict_to_time):
        """predict(self, ctrl_input, predict_to_time)
        Predict the state to the specified time given the control input
        (roll, pitch, yaw)
        """
        if self.last_time.predict == 0:
            self.last_time.predict = predict_to_time
            return
        delta_t = predict_to_time - self.last_time.predict
        self.last_time.predict = predict_to_time
        if delta_t < 0:
            return
        
        # Predict states - get transition matrices and process noise
        ((kf_trans_mat, kf_process_noise),
         (pf_trans_mat, pf_process_noise))= (
             self.trans_matrices(ctrl_input, delta_t))
        
        vehicle = self.vehicle
        # Kalman filter prediction
        # Predict the velocity - constant velocity model
        pred_kf_state = np.dot(kf_trans_mat, vehicle.kf_state)
        pred_kf_state_cov = np_kf_predict_cov(
            vehicle.kf_state_cov, kf_trans_mat, kf_process_noise)
        
        # Assume constant position model for first half of particles and
        # constant velocity model for the other half
        # Particle filter prediction
        # Draw noise samples from the Kalman filter distribution (covariance)
        kf_noise_samples = np.random.multivariate_normal(
            np.zeros(vehicle.kf_state.shape[0]), pred_kf_state_cov*self.vars.prediction_noise_scaling, self.vars.nparticles)
        #position_noise = kf_noise_samples[:, :3]
        # Generate samples for the velocity
        velocity_samples = kf_noise_samples[:, 3:]
        velocity_samples += pred_kf_state[3:]
        
        # Jump Markov model
        # Sample the mode (constant position/velocity)
        if self.vars.use_jms:
            mode = np.random.random(self.vars.nparticles)
            mode[mode > self.vars.imm_ratio] = 1
            mode[mode < 1] = 0
            mode += vehicle.mode
            mode = mode % 2
            vehicle.mode = mode
        cp_nparticles = np.where(vehicle.mode == 0)[0]
        
        # Set velocity_samples to zero for mode=0
        velocity_samples[cp_nparticles] = 0
        # Get the rotation matrix
        rot_mat = pf_trans_mat[:, 3:]
        # Predict according to motion model
        pred_position = np.asarray(
            vehicle.position + np.dot(rot_mat, velocity_samples.T).T, order='C')
        # Generate noise
        position_noise = np.random.multivariate_normal(
            np.zeros(3), pf_process_noise, self.vars.nparticles)
        pred_position += position_noise
        
        # Save prediction
        vehicle.kf_state = pred_kf_state
        vehicle.kf_state_cov = pred_kf_state_cov
        vehicle.position = pred_position
        
        # Copy the predicted states to the "parent state" attribute and 
        # perform a prediction for the map
        self._copy_state_to_map_(pred_position, ctrl_input, rot_mat)
    
    def _filter_update_(self, h_mat, r_mat, z, INPLACE=True):
        # Kalman filter update for the velocity
        # predicted observations
        vehicle = self.vehicle
        # Predicted velocity measurements
        pred_z = np.array(np.dot(h_mat, vehicle.kf_state.T).T, order='C')
        
        if misctools.mahalanobis(z, np.asarray(r_mat[np.newaxis], order='C'), 
                       np.asarray(pred_z[np.newaxis], order='C') ) <= 9:
            PERFORM_UPDATE = True
            # Updated Kalman filter covariance
            upd_kf_state_cov, kalman_info = np_kf_update_cov(vehicle.kf_state_cov,
                                                             h_mat, r_mat, INPLACE)
            # Predicted state
            pred_kf_state = vehicle.kf_state.copy()
            # Update the states
            upd_kf_state, residuals = np_kf_update_x(vehicle.kf_state, pred_z, z,
                kalman_info.kalman_gain, INPLACE)
        else:
            PERFORM_UPDATE = False
            print "RBPHDSLAM2:_FILTER_UPDATE_():"
            print "Mahalanobis distance exceeded, not performing update"
        
        if not INPLACE:
            if not PERFORM_UPDATE:
                upd_kf_state = vehicle.kf_state.copy()
                upd_kf_state_cov = vehicle.kf_state_cov.copy()
            return vehicle.weights, upd_kf_state, upd_kf_state_cov
        else:
            if PERFORM_UPDATE:
                delta_kf_state = upd_kf_state - pred_kf_state
                # Correct the position of particles with CV model
                cv_nparticles = np.where(self.vehicle.mode == 1)[0]
                vehicle.position[cv_nparticles] += delta_kf_state[:3]
            return None
    
    def update_gps(self, gps):
        pass
    
    def update_dvl(self, dvl, mode):
        """update_dvl(self, dvl, mode)
        Update the vehicle state using the dvl measurement.
        mode = 'b' or 'w' for bottom or water lock
        """
        self.flags.ESTIMATE_IS_VALID = False
        assert mode in ['b', 'w'], "Specify (b)ottom or (w)ater for dvl update"
        if mode == 'b':
            r_mat = self.vars.dvl_b_R #np.array([self.vars.dvl_b_R])
        else:
            r_mat = self.vars.dvl_w_R #np.array([self.vars.dvl_w_R])
        h_mat = self.vars.dvlH #np.array([self.vars.dvlH])
        self._filter_update_(h_mat, r_mat, dvl, INPLACE=True)
        # Copy the particle state to the PHD parent state
        self._copy_state_to_map_(self.vehicle.position)
    
    def update_svs(self, svs):
        """update_svs(self, svs)
        Update the state using the depth reading
        """
        self.flags.ESTIMATE_IS_VALID = False
        self.vehicle.position[:, 2] = svs
        # Copy the particle state to the PHD parent state
        self._copy_state_to_map_(self.vehicle.position)
    
    def update_features(self, features):
        """update_features(self, features)
        Update the map using the features. Features are specified as a Nx6
        numpy array where the first 3 columns are x, y, z positions and the
        last 3 columns are the diagonal of the covariance matrix for that
        measurement
        """
        self.flags.ESTIMATE_IS_VALID = False
        if features.shape[0]:
            features_pos = features[:, 0:3].copy()
            features_noise = np.array([np.diag(features[i, 3:6]) 
                for i in range(features.shape[0])])
        else:
            features_pos = np.empty((0, 3))
            features_noise = np.empty((0, 3, 3))
        #fig = plt.figure(4)
        #ax = fig.gca(projection="3d")
        #ax.cla()
        #ax.scatter3D(self.vehicle.position[:, 0], self.vehicle.position[:, 1], self.vehicle.weights)
        #ax.set_title("Before Update")
        slam_info = [self.vehicle.maps[i].iterate(features_pos, features_noise) 
            for i in range(self.vars.nparticles)]
        #slam_info = Parallel(n_jobs=2)(delayed(_update_features_parallel_)(self, idx, features_pos, features_noise) for idx in range(self.vars.nparticles))
        print "map size: ", self.vehicle.maps[0].weights.shape[0]
        # Overflow errors
        #slam_weight_update = np.array([slam_info[i].likelihood
        #    for i in range(self.vars.nparticles)])
        #self.vehicle.weights *= slam_weight_update/slam_weight_update.sum()
        #self.vehicle.weights /= self.vehicle.weights.sum()
        # Use log to avoid overflows
        slam_weight_log_update = np.array([slam_info[i].log_likelihood
            for i in range(self.vars.nparticles)])
        log_weights = np.log(self.vehicle.weights)
        slam_weight_log_update -= slam_weight_log_update.max()
        slam_weight_log_update_sum = np.log(np.sum(np.exp(slam_weight_log_update)))
        log_weights += slam_weight_log_update - slam_weight_log_update_sum
        self.vehicle.weights = np.exp(log_weights)
        self.vehicle.weights /= self.vehicle.weights.sum()
        #fig = plt.figure(5)
        #ax = fig.gca(projection="3d")
        #ax.cla()
        #ax.scatter3D(self.vehicle.position[:, 0], self.vehicle.position[:, 1], self.vehicle.weights)
        #ax.set_title("After Update")
        #code.interact(local=locals())
        
        #if features_pos.shape[0]:
        #    [self.vehicle.maps[i].birth(features_pos, features_noise, APPEND=True)
        #        for i in range(self.vars.nparticles)]
    
    def compress_maps(self, *args, **kwargs):
        """compress_maps(self, *args, **kwargs)
        Merge all possible landmarks in the maps
        """
        [_map_.merge_fov(-1) for _map_ in self.vehicle.maps]
    
    def estimate(self):
        """estimate(self) -> estimate
        Generate the state and map estimates
        """
        if not self.flags.ESTIMATE_IS_VALID:
            vehicle = self.vehicle
            position, cov = misctools.sample_mn_cv(vehicle.position, 
                                                   vehicle.weights)
            self._estimate_.vehicle.ned = SAMPLE(1, position, cov)
            max_weight_idx = vehicle.weights.argmax()
            self._estimate_.map = vehicle.maps[max_weight_idx].estimate()
            self.flags.ESTIMATE_IS_VALID = True
        return self._estimate_
    
    def resample(self):
        """resample(self)
        Resample the particles according to self.vars.resample_threshold
        """
        nparticles = self.vars.nparticles
        vehicle = self.vehicle
        # Effective number of particles
        eff_nparticles = 1/np.power(self.vehicle.weights, 2).sum()
        resample_threshold = eff_nparticles/nparticles
        # Check if we have particle depletion
        if (resample_threshold > self.vars.resample_threshold):
            return
        print "Resampling"
        if np.any(np.isnan(vehicle.weights)):
            print "\n\n***  NaNs in weights  ***\n\n"
        # Scale the number of particles
        #self.vars._nparticles_ *= 1.0015
        #self.vars.nparticles = int(min((self.vars._nparticles_, 400)))
        nparticles = self.vars.nparticles
        # Otherwise we need to resample
        # Resample imm_ratio for CP model plus 1-imm_ratio for CV model
        if not self.vars.use_jms:
            cp_nparticles = int(np.floor(self.vars.imm_ratio*nparticles))
            cv_nparticles = nparticles - cp_nparticles
            cp_resample_index = misctools.get_resample_index(vehicle.weights, cp_nparticles)
            cv_resample_index = misctools.get_resample_index(vehicle.weights, cv_nparticles)
            resample_index = np.hstack((cp_resample_index, cv_resample_index))
        else:
            resample_index = misctools.get_resample_index(vehicle.weights, nparticles)
            vehicle.mode = vehicle.mode[resample_index]
        # self.states is a numpy array so the indexing operation forces a copy
        vehicle.weights = np.ones(nparticles, dtype=float)/nparticles
        vehicle.position = np.asarray(vehicle.position[resample_index], order='C')
        
        # Only resample the maps if they are populated
        num_map_components = np.asarray(
            [_map_.weights.shape[0] for _map_ in vehicle.maps])
        if np.any(num_map_components):
            resampled_maps = [self.vehicle.maps[i].copy() for i in resample_index]
            # Map copies will contain references to the previous frame id.
            # Reinitialise to unique ids
            for count in range(nparticles):
                resampled_maps[count].sensors.camera.set_tf_frame(
                    *('slam_sensor'+str(count), 'slam_sensor_right'+str(count)))
                        #*self.vehicle.maps[count].sensors.camera.get_tf_frame())
            vehicle.maps = resampled_maps
