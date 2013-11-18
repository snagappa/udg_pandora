# -*- coding: utf-8 -*-
#cython: wraparound=False, boundscheck=False

import numpy as np
cimport numpy as np
#from scipy.lib.lapack.clapack import dpotrf, dpotri


def mahalanobis_distance_sq(np.ndarray[double, ndim=1] x not None,
                         np.ndarray[double, ndim=2] P not None,
                         np.ndarray[double, ndim=2] y not None):
    cdef np.ndarray[double, ndim=2] invP #= P.copy()
    cdef np.ndarray[double, ndim=2] invP_times_residual
    #dpotrf(invP, "overwrite_a")
    #dpotri(invP, "overwrite_c")
    invP = np.linalg.inv(P)
    residual = x-y
    invP_times_residual = np.dot(invP, residual.T).T
    md_sq = (residual*invP_times_residual).sum(axis=1)
    return md_sq

def approximate_mahalanobis_distance_sq(np.ndarray[double, ndim=1] x not None,
                                     np.ndarray[double, ndim=2] P not None,
                                     np.ndarray[double, ndim=2] y not None):
    cdef np.ndarray[double, ndim=1] invP = 1/P.diagonal()
    cdef np.ndarray[double, ndim=2] invP_times_residual
    residual = x-y
    invP_times_residual = invP*residual
    md_sq = (residual*invP_times_residual).sum(axis=1)
    return md_sq

def merge_states(np.ndarray[double, ndim=1] wt not None,
                 np.ndarray[double, ndim=2] x not None,
                 np.ndarray[double, ndim=3] P not None):
    """
    Compute the weighted mean and covariance from a (numpy) list of states and
    covariances with their weights.
    """
    cdef double merged_wt = wt.sum()
    cdef np.ndarray[double, ndim=1] merged_x = (wt[:, np.newaxis]*x).sum(0)/merged_wt
    cdef np.ndarray[double, ndim=3] residual = (x - merged_x)[:, np.newaxis]
    cdef np.ndarray[double, ndim=3] P_copy
    P_copy = P + [np.dot(residual[i].T, residual[i]) for i in range(residual.shape[0])]
    merged_P = (wt[:, np.newaxis, np.newaxis]*P_copy).sum(axis=0)/merged_wt
    return merged_wt, merged_x, merged_P

def rfs_merge(np.ndarray[double, ndim=1] weights not None,
          np.ndarray[double, ndim=2] states not None,
          np.ndarray[double, ndim=3] covs not None,
          double merge_threshold, APPROXIMATE_DISTANCE=True,
          double weight_threshold=0.3):
    cdef int num_remaining_components = weights.shape[0]
    cdef unsigned int max_wt_index
    cdef np.ndarray[double, ndim=1] max_wt_state
    cdef np.ndarray[double, ndim=2] max_wt_cov
    
    cdef np.ndarray[double, ndim=1] merged_wts = np.empty(0)
    cdef np.ndarray[double, ndim=2] merged_sts = np.empty((0, states.shape[1]))
    cdef np.ndarray[double, ndim=3] merged_cvs = np.empty(
        (0, states.shape[1], states.shape[1]))
    if APPROXIMATE_DISTANCE:
        mahalanobis_fn = approximate_mahalanobis_distance_sq
    else:
        mahalanobis_fn = mahalanobis_distance_sq
    # Calculate the squared threshold
    merge_threshold_sq = merge_threshold**2
    while num_remaining_components:
        max_wt_index = weights.argmax()
        # Only perform merge if 
        if weights[max_wt_index] < weight_threshold:
            merged_wts = np.hstack((merged_wts, weights))
            merged_sts = np.vstack((merged_sts, states))
            merged_cvs = np.vstack((merged_cvs, covs))
            break
            
        max_wt_state = states[max_wt_index]
        max_wt_cov = covs[max_wt_index]
        mahalanobis_dist_sq = mahalanobis_fn(max_wt_state, max_wt_cov, states)
        merge_list_indices_logical = mahalanobis_dist_sq <= merge_threshold_sq
        retain_idx_logical = mahalanobis_dist_sq > merge_threshold
        new_wt, new_st, new_cv = merge_states(weights[merge_list_indices_logical], 
                                              states[merge_list_indices_logical],
                                              covs[merge_list_indices_logical])
        merged_wts = np.hstack((merged_wts, new_wt))
        merged_sts = np.vstack((merged_sts, new_st))
        merged_cvs = np.vstack((merged_cvs, new_cv[np.newaxis]))
        # Remove merged states from the list
        weights = weights[retain_idx_logical]
        states = states[retain_idx_logical]
        covs = covs[retain_idx_logical]
        num_remaining_components = weights.shape[0]
    
    return merged_wts, merged_sts, merged_cvs
