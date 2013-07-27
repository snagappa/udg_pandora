# -*- coding: utf-8 -*-

USE_CYTHON = False


import numpy as np
import misctools, blas
#from featuredetector import sensors, tf
import cameramodels
from kalmanfilter import kf_predict_cov, np_kf_update_cov, kf_update_cov, kf_update_x
from collections import namedtuple
import copy
import threading
from misctools import STRUCT, rotation_matrix
import sys
import traceback

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
    def __init__(self, prune_threshold=1e-3, merge_threshold=1,
                 max_num_components=4000, birth_intensity=0.05,
                 clutter_intensity=10, detection_probability=0.8):
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
        
        self._estimate_ = SAMPLE(np.zeros(0), 
                                 np.zeros((0, 3)), np.zeros((0, 3, 3)))
        
        self.vars = STRUCT()
        # Prune components less than this weight
        self.vars.prune_threshold = float(prune_threshold)
        # Merge components  closer than this threshold
        self.vars.merge_threshold = float(merge_threshold)
        # Maximum number of components to track
        self.vars.max_num_components = int(max_num_components)
        # Intensity of new targets
        self.vars.birth_intensity = float(birth_intensity)
        # Intensity of clutter in the scene
        self.vars.clutter_intensity = float(clutter_intensity)
        # Probability of detection of targets in the FoV
        self.vars.pd = detection_probability
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = False
        self.flags.LOCK = threading.RLock()
        
        self.sensors = STRUCT()
        try:
            self.sensors.camera = cameramodels.StereoCameraModel()
        except:
            print "Error initialising camera models"
            exc_info = sys.exc_info()
            print "GMPHD:__INIT__():\n", traceback.print_tb(exc_info[2])
            
        self.sensors.camera.set_const_pd(self.vars.pd)
    
    def copy(self):
        """phd.copy() -> phd_copy
        Create a new copy of the GMPHD object"""
        new_object = GMPHD()
        self.flags.LOCK.acquire()
        try:
            new_object.weights = self.weights.copy()
            new_object.states = self.states.copy()
            new_object.covs = self.covs.copy()
            new_object.parent_ned = self.parent_ned.copy()
            new_object.parent_rpy = self.parent_rpy.copy()
            new_object._estimate_ = self.estimate()
            new_object.vars = copy.copy(self.vars)
            new_object.flags = copy.copy(self.flags)
            new_object.flags.LOCK = threading.RLock()
            new_object.sensors.camera = copy.deepcopy(self.sensors.camera)
        except:
            exc_info = sys.exc_info()
            print "GMPHD:COPY():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
        return new_object
    
    def set_states(self, ptr_weights, ptr_states, ptr_covs):
        """phd.set_states(ptr_weights, ptr_states, ptr_covs)
        Assign new weights, states and covariances to the object. A copy of the
        arguments is not made - changes to the contents of the original
        variables in-place will be visible inside the object.
        """
        self.flags.LOCK.acquire()
        try:
            self.flags.ESTIMATE_IS_VALID = False
            self.weights = ptr_weights
            self.states = ptr_states
            self.covs = ptr_covs
        except:
            exc_info = sys.exc_info()
            print "GMPHD:SET_STATES():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
    
    def set_parent(self, parent_ned, parent_rpy):
        """phd.set_parent(parend_ned, parent_rpy)
        Set new values of the parent state
        parent_ned - numpy array indicating position as (north, east, down)
        parent_rpy - numpy array indicating orientation as (roll, pitch, yaw)
        """
        self.flags.LOCK.acquire()
        try:
            self.parent_ned = parent_ned.copy()
            self.parent_rpy = parent_rpy.copy()
        except:
            exc_info = sys.exc_info()
            print "GMPHD:SET_PARENT():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
    
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
        slam_info = STRUCT()
        slam_info.likelihood = 1.
        num_observations, z_dim = (observations.shape + (3,))[0:2]
        
        if not self.weights.shape[0]:
            #print "nothing to update"
            return slam_info
        camera = self.sensors.camera
        detection_probability = camera.pdf_detection(self.states, margin=0)
        clutter_pdf = camera.pdf_clutter(observations)
        clutter_intensity = self.vars.clutter_intensity*clutter_pdf
        
        self.flags.LOCK.acquire()
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
            slam_info.exp_sum__pd_predwt = np.exp(-detected.weights.sum())
            
            # SLAM, prep for step 2:
            slam_info.sum__clutter_with_pd_updwt = np.zeros(num_observations)
            
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
                    observation_noise = observation_noise[0]
                    detected.covs, kalman_info = kf_update_cov(
                        detected.covs, h_mat, observation_noise, INPLACE=True)
                # We need to update the states and find the updated weights
                for (_observation_, obs_count) in zip(observations, 
                                                      range(num_observations)):
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
                    
                    # Normalise the weights
                    normalisation_factor = ( clutter_intensity[obs_count] + 
                                             #self.vars.birth_intensity +
                                             upd_weights.sum() )
                    upd_weights /= normalisation_factor
                    #print "Obs Index: ", str(obs_count+1)
                    #print upd_weights.sum()
                    # SLAM, step 2:
                    slam_info.sum__clutter_with_pd_updwt[obs_count] = \
                        normalisation_factor
                    
                    # Create new state with new_x and P to add to _states_
                    updated.weights += [upd_weights.copy()]
                    updated.states += [upd_states.copy()]
                    updated.covs += [detected.covs.copy()]
                    #print upd_weights.sum()
            else:
                slam_info.sum__clutter_with_pd_updwt = np.ones(num_observations)
            
            self.weights = np.concatenate(updated.weights)
            self.states = np.concatenate(updated.states)
            self.covs = np.concatenate(updated.covs)
            
            # SLAM, finalise:
            slam_info.likelihood = (slam_info.exp_sum__pd_predwt * 
                                    slam_info.sum__clutter_with_pd_updwt.prod())
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            print "error in update"
            exc_info = sys.exc_info()
            print "GMPHD:UPDATE():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
        return slam_info
    
    def estimate(self):
        """phd.estimate -> (weights, states, covariances)
        Estimate the number of landmarks in the map and return as a tuple of
        the weights, states and covariances of the landmarks.
        """
        if not self.flags.ESTIMATE_IS_VALID:
            self.flags.LOCK.acquire()
            try:
                weights = self.weights.copy()
                states = self.states.copy()
                covs = self.covs.copy()
            except:
                exc_info = sys.exc_info()
                print "GMPHD:ESTIMATE():\n", traceback.print_tb(exc_info[2])
            finally:
                self.flags.LOCK.release()
            valid_targets = weights>0.5
            num_targets = valid_targets.sum()
            
            #intensity = weights.sum()
            #num_targets = min((intensity, weights.shape[0]))
            #num_targets = int(round(num_targets))
            if num_targets:
                #inds = np.flipud(weights.argsort())
                #inds = inds[0:num_targets]
                #est_weights = weights[inds]
                #est_states = states[inds]
                #est_covs = covs[inds]
                ## Discard states with low weight
                #valid_idx = np.where(est_weights>0.4)[0]
                #self._estimate_ = SAMPLE(est_weights[valid_idx],
                #                         est_states[valid_idx], 
                #                         est_covs[valid_idx])
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
    
    def prune(self, override_prune_threshold=None, 
              override_max_num_components=None):
        """phd.prune()
        Remove landmarks in the map with low weights.
        """
        self.flags.LOCK.acquire()
        try:
            # Get the threshold for weight based pruning
            prune_threshold = self.vars.prune_threshold
            if not override_prune_threshold is None:
                prune_threshold = override_prune_threshold
            max_num_components = self.vars.max_num_components
            if not override_max_num_components is None:
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
            if self.weights.shape[0] > max_num_components:
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
        finally:
            self.flags.LOCK.release()
    
    def merge_fov(self, detection_threshold=0.1):
        """phd.merge_fov(detection_threshold=0.5)
        Merge Gaussian components which are in the field of view or which
        satisfy a probability of detection given by detection_threshold.
        Set detection_threshold to 0 to merge landmarks everywhere in the map.
        """
        self.flags.LOCK.acquire()
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
            num_remaining_components = self.weights.shape[0]
            
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
            
            self.flags.ESTIMATE_IS_VALID = False
            self.set_states(np.hstack((unmerged_wts, np.array(merged_wts))), 
                            np.vstack((unmerged_sts, np.array(merged_sts))), 
                            np.vstack((unmerged_cvs, np.array(merged_cvs))))
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            exc_info = sys.exc_info()
            print "GMPHD:MERGE_FOV():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
    
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
        self.flags.LOCK.acquire()
        try:
            self.flags.ESTIMATE_IS_VALID = False
            self.weights = np.hstack((self.weights, weights))
            if DEBUG:
                assert len(states.shape) == 2, "states must be a nxm ndarray"
                assert len(covs.shape) == 3, "covs must be a nxmxm ndarray"
            self.states = np.vstack((self.states, states))
            self.covs = np.vstack((self.covs, covs))
            assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        except:
            exc_info = sys.exc_info()
            print "GMPHD:APPEND():\n", traceback.print_tb(exc_info[2])
        finally:
            self.flags.LOCK.release()
    
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
        self.prune(override_prune_threshold=self.vars.prune_threshold/10, 
                   override_max_num_components=np.inf)
        # Add birth terms
        if observations.shape[0]:
            self.birth(observations, obs_noise, APPEND=True)
        # Merge states in the field of view - too expensive to merge whole
        # state space
        self.merge_fov()
        # Generate vehicle state and map estimates
        self.estimate()
        # Perform normal prune
        self.prune()
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
            visible_idx = camera.is_visible_relative2sensor(features_rel, 
                None, margin=-0.18)
            birth_features = features_rel[visible_idx]
            features_cv = features_cv[visible_idx]
            if birth_features.shape[0]:
                birth_wt = self.vars.birth_intensity*np.ones(birth_features.shape[0])
                try:
                    birth_st = self.sensors.camera.to_world_coords(birth_features)
                except:
                    print "tf conversion to world coords failed!"
                    exc_info = sys.exc_info()
                    print "GMPHD:CAMERA_BIRTH():\n", traceback.print_tb(exc_info[2])
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
