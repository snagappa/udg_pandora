# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:14:38 2012

@author: snagappa
"""

import blas
import numpy as np

#class kalmanfilter(object): pass
__all__ = []

def kf_predict(state, covariance, F, Q, B=None, u=None):
    pred_state = blas.dgemv(F, state)
    if (not B==None) and (not u==None):
        blas.dgemv(B, u, beta=np.array([1.0]), y=pred_state)
    # Repeat Q n times and return as the predicted covariance
    pred_cov = np.repeat(np.array([Q]), state.shape[0], 0)
    blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True), 
               beta=np.array([1.0]), C=pred_cov)
    return pred_state, pred_cov
    

def kf_update(state, covariance, H, R, z=None, INPLACE=True):
    kalman_info = lambda:0
    assert z == None or len(z.shape) == 1, "z must be a single observations, \
    not an array of observations"
    
    if INPLACE:
        upd_state = state
    else:
        upd_state = state.copy()
    # Update the covariance and generate the Kalman gain, etc.
    upd_covariance, kalman_info = kf_update_cov(covariance, H, R, INPLACE)
    
    # Observation from current state
    pred_z = blas.dgemv(H, state)
    if not (z==None):
        upd_state, residuals = kf_update_x(upd_state, pred_z, z, 
                                           kalman_info.kalman_gain, INPLACE=True)
    else:
        residuals = np.empty(0)
    
    kalman_info.pred_z = pred_z
    kalman_info.residuals = residuals
    
    return upd_state, upd_covariance, kalman_info
    
    
def kf_update_x(x, pred_z, z, kalman_gain, INPLACE=True):
    #assert len(z.shape) == 1, "z must be a single observations, \
    #not an array of observations"
    if INPLACE:
        upd_state = x
    else:
        upd_state = x.copy()
    
    #residuals = np.repeat([z], pred_z.shape[0], 0)
    #blas.daxpy(-1, pred_z, residuals)
    residuals = np.asanyarray(z - pred_z, order='C')
    # Update the state
    try:
        blas.dgemv(kalman_gain, residuals, beta=np.array([1.0]), y=upd_state)
    except AssertionError as assert_err:
        print assert_err
        raise RuntimeError("Error in kf_update_x")
    
    return upd_state, residuals
    

def np_kf_update_x(x, pred_z, z, kalman_gain, INPLACE=True):
    assert len(z.shape) == 1, "z must be a single observations, \
    not an array of observations"
    if INPLACE:
        upd_state = x
    else:
        upd_state = x.copy()
    
    #residuals = np.repeat([z], pred_z.shape[0], 0)
    #blas.daxpy(-1, pred_z, residuals)
    residuals = z - pred_z
    # Update the state
    upd_state += np.dot(kalman_gain, residuals.T).T
    
    return upd_state, residuals
    
def np_kf_predict_cov(covariance, F, Q):
    pred_cov = np.dot(F, np.dot(covariance, F.T)) + Q
    return pred_cov

def kf_predict_cov(covariance, F, Q):
    # Repeat Q n times and return as the predicted covariance
    #pred_cov = np.repeat(np.array([Q]), covariance.shape[0], 0)
    #blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True), 
    #           beta=np.array([1.0]), C=pred_cov)
    pred_cov = blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True)) + Q
    return pred_cov
    
def np_kf_update_cov(covariance, H, R, INPLACE=True):
    kalman_info = lambda:0
    
    if INPLACE:
        upd_covariance = covariance
        covariance_copy = covariance.copy()
    else:
        upd_covariance = covariance.copy()
        covariance_copy = covariance
    
    # Comput PH^T
    p_ht = np.dot(covariance, H.T)
    # Compute HPH^T + R
    hp_ht_pR = np.dot(H, p_ht) + R
    
    # Compute the Cholesky decomposition
    chol_S = np.linalg.cholesky(hp_ht_pR)
    # Compute the determinant
    det_S = (np.diag(chol_S).prod())**2
    # Compute the inverse of the square root
    inv_sqrt_S = np.array(np.linalg.inv(chol_S), order='C')
    # and the inverse
    inv_S = np.dot(inv_sqrt_S.T, inv_sqrt_S)
    
    # Kalman gain
    kalman_gain = np.dot(p_ht, inv_S)
    
    # Update the covariance
    k_h = np.dot(kalman_gain, H)
    upd_covariance -= np.dot(k_h, covariance_copy)
    
    kalman_info.S = hp_ht_pR
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.kalman_gain = kalman_gain
    
    return upd_covariance, kalman_info
    
    
def kf_update_cov(covariance, H, R, INPLACE=True):
    kalman_info = lambda:0
    
    if INPLACE:
        upd_covariance = covariance
        covariance_copy = covariance.copy()
    else:
        upd_covariance = covariance.copy()
        covariance_copy = covariance
    
    # Store R
    #chol_S = np.repeat(R, covariance.shape[0], 0)
    # Compute PH^T
    p_ht = blas.dgemm(covariance, H, TRANSPOSE_B=True)
    # Compute HPH^T + R
    #blas.dgemm(H, p_ht, C=chol_S)
    hp_ht_pR = blas.dgemm(H, p_ht) + R
    # Compute the Cholesky decomposition
    chol_S = blas.dpotrf(hp_ht_pR, False)
    # Select the lower triangle (set the upper triangle to zero)
    blas.mktril(chol_S)
    # Compute the determinant
    diag_vec = np.array([np.diag(chol_S[i]) for i in range(chol_S.shape[0])])
    det_S = diag_vec.prod(1)**2
    # Compute the inverse of the square root
    inv_sqrt_S = blas.dtrtri(chol_S, 'l')
    # Compute the inverse using dsyrk
    inv_S = blas.dsyrk('l', inv_sqrt_S, TRANSPOSE_A=True)
    # Symmetrise the matrix since only the lower triangle is stored
    blas.symmetrise(inv_S, 'l')
    #blas.dpotri(op_S, True)
    # inv_S = op_S
    
    # Kalman gain
    kalman_gain = blas.dgemm(p_ht, inv_S)
    
    # Update the covariance
    k_h = blas.dgemm(kalman_gain, H)
    blas.dgemm(k_h, covariance_copy, alpha=np.array([-1.0]), 
               beta=np.array([1.0]), C=upd_covariance)
    
    kalman_info.S = hp_ht_pR
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.kalman_gain = kalman_gain
    
    return upd_covariance, kalman_info
    

def ukf_predict(states, covs, ctrl_input, proc_noise, predict_fn, 
                delta_t, parameters, _alpha=1e-3, _beta=0, _kappa=0):
    # Time update
    covs += proc_noise# + 1e-3*np.eye(covs.shape[0])
    sigma_x, wt_mn, wt_cv = sigma_pts(states, covs, _alpha, _beta, _kappa)
    
    sigma_x_pred = predict_fn(sigma_x, ctrl_input, delta_t, 
                                                parameters)
    
    # Predicted state is weighted mean of predicted sigma points
    pred_state = sigma_x_pred.copy()
    #blas.dscal(wt_mn, pred_state)
    pred_state *= wt_mn[:, np.newaxis]
    pred_state = pred_state.sum(axis=0)
    
    # Predicted covariance is weighted mean of sigma covariance + proc_noise
    pred_cov = _sigma_cov_(sigma_x_pred, pred_state, wt_cv, 0)
    return pred_state, pred_cov


def np_ukf_update_cov(state, cov, obs_noise, obsfn, obsfn_args=(), _alpha=1e-3, _beta=2, _kappa=0, INPLACE=True):
    assert state.ndim == 1, "state must be 1D"
    assert cov.ndim == 2, "cov must be 2D"
    assert obs_noise.ndim == 2, "obs_noise must be 2D"
    
    kalman_info = lambda:0
    
    state_dim = state.shape[0]
    num_sigma_pts = 2*state_dim + 1
    
    # Generate the sigma points
    (x_sigma, wt_mn, wt_cv) = np_sigma_pts(state, cov, _alpha, _beta, _kappa)
    # predicted mean x - same as current x?
    predicted_x = (wt_mn[:, np.newaxis]*x_sigma).sum(axis=0)
    
    # Generate observations for predicted sigma points
    z_sigma = np.ascontiguousarray(obsfn(x_sigma, *obsfn_args))
    # predicted mean z
    predicted_z = (wt_mn[:, np.newaxis]*z_sigma).sum(axis=0)
    
    # Innovation covariance - used to compute the likelihood
    z_residual = z_sigma - predicted_z
    zz_sigma_cov = np.asarray(
        [wt_cv[count]*np.dot(z_residual[[count]].T, z_residual[[count]]) 
        for count in range(num_sigma_pts)]).sum(axis=0) + obs_noise
    
    x_residual = x_sigma - predicted_x
    xz_sigma_cov = np.asarray(
        [wt_cv[count]*np.dot(x_residual[[count]].T, z_residual[[count]]) 
        for count in range(num_sigma_pts)]).sum(axis=0)
    
    # Kalman gain
    kalman_gain = np.dot(xz_sigma_cov, np.linalg.inv(zz_sigma_cov))
    
    cov_update_term = np.dot(kalman_gain, np.dot(zz_sigma_cov, kalman_gain.T))
    if INPLACE:
        cov -= cov_update_term
    else:
        cov = cov - cov_update_term
    
    kalman_info.S = zz_sigma_cov
    kalman_info.kalman_gain = kalman_gain
    return cov, predicted_z, kalman_info


def ukf_update_cov(states, covs, obs_noise, obsfn, obsfn_args=(), _alpha=1e-3, _beta=2, _kappa=0, INPLACE=True):
    assert states.ndim == 2, "states must be 2D"
    assert covs.ndim == 3, "covs must be 3D"
    assert 2 <= obs_noise.ndim <= 3, "obs_noise must be 2D or 3D"
    
    kalman_info = lambda:0
    
    num_states, state_dim = states.shape
    num_sigma_pts = 2*state_dim + 1
    
    # Redraw sigma points
    (x_sigma, x_weight, p_weight) = sigma_pts(states, covs, _alpha, _beta, _kappa)
    # predicted mean x - same as current x?
    predicted_x = (x_weight[np.newaxis, :, np.newaxis]*x_sigma).sum(axis=1)
    
    # Generate observations for predicted sigma points
    x_sigma_flat = np.reshape(x_sigma, (num_states*num_sigma_pts, x_sigma.shape[2]))
    z_sigma_flat = obsfn(x_sigma_flat, *obsfn_args)
    z_dim = z_sigma_flat.shape[1]
    z_sigma = np.reshape(z_sigma_flat, (num_states, num_sigma_pts, z_dim))
    predicted_z = (x_weight[np.newaxis, :, np.newaxis]*z_sigma).sum(axis=1)
    
    # Observation covariance
    z_diff_flat = np.asarray(np.reshape(z_sigma - predicted_z[:, np.newaxis, :],
                             (num_states*num_sigma_pts, z_dim)), order='C')
    
    # Innovation covariance - used to compute the likelihood
    cov_wts = (p_weight * np.ones((num_states, 1))).ravel()
    zz_sigma_cov = np.asarray(
        np.reshape(blas.dsyr('l', z_diff_flat, cov_wts),
                   (num_states, num_sigma_pts, z_dim, z_dim)).sum(axis=1) + obs_noise, order='C')
    blas.symmetrise(zz_sigma_cov, 'l')
    
    # Compute cross covariance
    x_diff_flat = np.reshape(x_sigma - predicted_x[:, np.newaxis, :],
                             (num_states*num_sigma_pts, x_sigma.shape[2]))
    xz_sigma_cov = (
        np.reshape(blas.dger(x_diff_flat, z_diff_flat, cov_wts),
                   (num_states, num_sigma_pts, state_dim, z_dim)).sum(axis=1))
    
    # Kalman gain
    kalman_gain = blas.dgemm(xz_sigma_cov, blas.inverse(zz_sigma_cov, 'C', False))
    
    cov_update_term = blas.dgemm(kalman_gain, blas.dgemm(zz_sigma_cov, kalman_gain, TRANSPOSE_B=True))
    if INPLACE:
        covs -= cov_update_term
    else:
        covs = covs - cov_update_term
    #try:
    #    blas.dpotrf(covs, INPLACE=False)
    #except:
    #    print "Updated covariance is not positive definite"
    #    from IPython import embed
    #    embed()
    
    # Compute the Cholesky decomposition
    chol_S = blas.dpotrf(zz_sigma_cov, False)
    # Select the lower triangle (set the upper triangle to zero)
    blas.mktril(chol_S)
    # Compute the determinant
    diag_vec = np.array([np.diag(chol_S[i]) for i in range(chol_S.shape[0])])
    det_S = diag_vec.prod(1)**2
    # Compute the inverse of the square root
    inv_sqrt_S = blas.dtrtri(chol_S, 'l')
    
    kalman_info.S = zz_sigma_cov
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.kalman_gain = kalman_gain
    return covs, predicted_z, kalman_info
    

def np_sigma_pts(x, x_cov, _alpha=1e-3, _beta=2, _kappa=0):
    # State dimensions
    _L = x.shape[0]
    # UKF parameters
    _lambda = _alpha**2 * (_L+_kappa) - _L
    _gamma = np.sqrt(_L + _lambda)
    
    # Square root of scaled covariance matrix
    sqrt_cov = _gamma*np.linalg.cholesky(x_cov)
    
    # Array of the sigma points
    sigma_x = np.vstack((x, x+sqrt_cov.T, x-sqrt_cov.T))
    
    # Array of the weights for each sigma point
    wt_mn = 0.5*np.ones(1+2*_L)/(_L+_lambda)
    wt_mn[0] = _lambda/(_L+_lambda)
    wt_cv = wt_mn.copy()
    wt_cv[0] += (1 - _alpha**2 + _beta)
    return sigma_x, wt_mn, wt_cv

def _sigma_cov_(sigma_x, x_hat, wt_cv, proc_noise):
    residuals = sigma_x - x_hat
    sigma_cov = np.array([ (blas.dsyr('l', residuals, wt_cv)).sum(axis=0) ])
    blas.symmetrise(sigma_cov, 'l')
    sigma_cov = sigma_cov[0] + proc_noise

def sigma_pts(states, covs, _alpha=1e-3, _beta=2, _kappa=0):
    assert states.ndim == 2, "states must be 2D"
    assert covs.ndim == 3, "covs must be 3D"
    
    num_states, state_dim = states.shape
    _L = state_dim
    _lambda = _alpha**2*(_L + _kappa) - _L
    _L_plus_lambda = _L + _lambda
    
    # Vectors are stored row-wise and page-wise for each state-cov pair
    sigma_vecs = np.zeros((num_states, 2*state_dim+1, state_dim))
    sqrt_covs = blas.dpotrf(_L_plus_lambda*covs, INPLACE=False)
    # Use upper triangular since we want the transpose
    blas.mktriu(sqrt_covs)
    
    sigma_vecs[:, 0, :] = states
    sigma_vecs[:, 1:_L+1, :] = states[:, np.newaxis] + sqrt_covs
    sigma_vecs[:, _L+1:, :] = states[:, np.newaxis] - sqrt_covs
    wts_mn = 1./(2*_L_plus_lambda)*np.ones(2*state_dim+1)
    wts_mn[0] = _lambda/_L_plus_lambda
    wts_cv = wts_mn.copy()
    wts_cv[0] += (1 - _alpha**2 + _beta)
    return sigma_vecs, wts_mn, wts_cv
    


def evalSigmaCovariance(self, wt_vector, sigma_x1, x1, sigma_x2=None, x2=None):
    difference1 = [np.matrix(_sigma_x1 - x1) for _sigma_x1 in sigma_x1]
    if not (sigma_x2 is None):
        difference2 = [np.matrix(_sigma_x2 - x2) for _sigma_x2 in sigma_x2]
    else:
        difference2 = difference1
    
    sigma_cov = [this_wt_vector*_diff1.T*_diff2 for (this_wt_vector,_diff1, _diff2) in zip(wt_vector, difference1, difference2)]
    return np.add.reduce(sigma_cov)