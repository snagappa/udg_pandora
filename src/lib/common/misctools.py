#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       misctools.py
#       
#       Copyright 2012 Sharad Nagappa <snagappa@gmail.com>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#      

import blas
import numpy as np
from scipy import weave
from operator import mul, add
import code
import numpy as np

class STRUCT(object):
    def __init__(self):
        self.__objvars__ = dir(self)
        self.__objvars__ += ["__objvars__", "__dtype__"]
    def __repr__(self):
        lines = []
        allvars = dir(self)
        [lines.append(
            _var_ + " = " + str(getattr(self, _var_)) + self.__dtype__(getattr(self, _var_)))
            for _var_ in allvars if not _var_ in self.__objvars__]
        return "\n".join(lines)
    def __dtype__(self, var):
        dtype = ", " + str(type(var))
        if type(var) is np.ndarray:
            dtype += ", " + str(var.dtype)
        return dtype
        

    
def gen_retain_idx(array_len, del_idx):
    if del_idx is None or len(del_idx) == 0:
        return range(array_len)
    retain_idx = [i for i in range(array_len) if i not in del_idx]
    return retain_idx

def mahalanobis(x, P, y):
    """
    Compute the Mahalanobis distance given x, P and y as 
    sqrt((x-y)'*inv(P)*(x-y)).
    """
    residual = x-y
    if P.shape[0] == 1:
        p_times_residual = np.linalg.solve(P[0], residual.T).T
    else:
        p_times_residual, _ = blas.dposv(P, residual, OVERWRITE_A=False)
    distance = (residual*p_times_residual).sum(1)**0.5
    #blas_result = np.power(blas.ddot(residual, p_times_residual), 0.5)
    return distance
    
def approximate_mahalanobis(x, P, y):
    # Compute the mahalanobis distance using the diagonal of the matrix P
    assert P.shape[1] == P.shape[2], "P must be a square matrix"
    select_diag_idx = xrange(P.shape[1])
    residual = x-y
    diag_P = P[:, select_diag_idx, select_diag_idx]
    p_times_residual = (1.0/diag_P)*residual
    distance = (residual*p_times_residual).sum(1)**0.5
    return distance

def merge_states(wt, x, P):
    """
    Compute the weighted mean and covariance from a (numpy) list of states and
    covariances with their weights.
    """
    merged_wt = wt.sum()
    #merged_x = np.sum(blas.daxpy(wt, x), 0)/merged_wt
    merged_x = (wt[:, np.newaxis]*x).sum(0)/merged_wt
    """
    residual = x.copy()
    blas.daxpy(-1.0, np.array([merged_x]), residual)
    """
    residual = x - merged_x
    # method 1:
    # Convert the residual to a column vector
    #residual.shape += (1,)
    #P_copy = P.copy()
    #blas.dsyr('l', residual, 1.0, P_copy)
    #merged_P = np.array([(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], order='C')
    #blas.symmetrise(merged_P, 'l')
    
    # method 2:
    #P_copy = P + blas.dger(residual, residual)
    
    # method 3:
    P_copy = P + [residual[np.newaxis,i].T * residual[np.newaxis,i] 
        for i in xrange(residual.shape[0])]
    
    #merged_P = np.array([(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], order='C')
    merged_P = (wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt
    return merged_wt, merged_x, merged_P
    
def get_resample_index(weights, nparticles=-1):
    weights = weights/weights.sum()
    
    if nparticles==-1:
        nparticles = weights.shape[0]
    
    resampled_indices = np.empty(nparticles, dtype=int)
    wt_cdf = np.empty(weights.shape, dtype=float)
    u1 = np.random.uniform()/nparticles
    
    python_vars = ['nparticles', 'weights', 'wt_cdf', 'u1', 'resampled_indices']
    code = """
    double normfac = 0;
    int j = 0;
    int array_cur_loc = 0;
    double uj, d_u1;

    wt_cdf[0] = weights[0];
    for (j=1; j<Nweights[0]; j++)
        wt_cdf[j] = wt_cdf[j-1] + weights[j];
    
    for (j=0; j<nparticles; j++) {
        uj = u1 + (double)j/(double)nparticles;
        while (wt_cdf[array_cur_loc] < uj) {
            array_cur_loc++;
        }
        resampled_indices[j] = array_cur_loc;
    }
    """
    weave.inline(code, python_vars, extra_compile_args=["-O3"])
    return resampled_indices
    
    
def mvnpdf(x, mu, sigma):
    # Compute the residuals
    #if x.shape[0] == 1:
    #    residual = np.repeat(x, mu.shape[0], 0)
    #else:
    #    residual = x.copy(order='c')
    #blas.daxpy(-1.0, mu, residual)
    residual = x-mu
    #if x.shape[0] == 1:
    #    x = np.repeat(x, mu.shape[0], 0)
    #residual = x-mu
    chol_sigma = blas.dpotrf(sigma)
    # Compute the determinant
    diag_vec = np.array([np.diag(chol_sigma[i]) 
                        for i in range(chol_sigma.shape[0])])
    det_sigma = diag_vec.prod(1)**2
    
    # If same number of sigma and residuals, or only residual and many sigma
    if sigma.shape[0] == residual.shape[0] or residual.shape[0] == 1:
        inv_sigma_times_residual = blas.dpotrs(chol_sigma, residual)
        exp_term = blas.ddot(residual, inv_sigma_times_residual)
        
    # Otherwise, we have only one sigma - compute the inverse once
    else:
        # Compute the inverse of the square root
        inv_sqrt_sigma = blas.dtrtri(chol_sigma, 'l')
        exp_term = np.power(blas.dgemv(inv_sqrt_sigma,residual), 2).sum(axis=1)
    
    pdf = np.exp(-0.5*exp_term)/np.sqrt(det_sigma*(2*np.pi)**residual.shape[1])
    return pdf
    
    
def sample_mn_cv(x, wt=None, SYMMETRISE=False):
    if wt.shape[0] == 1:
        return x[0].copy(), np.zeros((x.shape[1], x.shape[1]))
    if wt==None:
        wt = 1.0/x.shape[0]*np.ones(x.shape[0])
    else:
        wt /= wt.sum()
    
    #mean_x = x.copy()
    # Scale the state by the associated weight
    #blas.dscal(wt, mean_x)
    # and take the sum
    #mean_x = mean_x.sum(axis=0)
    
    #residuals = x.copy()
    #blas.daxpy(-1.0, np.array([mean_x]), residuals)
    
    #mean_x = np.apply_along_axis(mul, 0, x, wt).sum(axis=0)
    mean_x = (wt[:,np.newaxis]*x).sum(axis=0)
    residuals = x - mean_x
    cov_x = np.array([blas.dsyr('l', residuals, wt).sum(axis=0)/(1-(wt**2).sum())])
    if SYMMETRISE:
        blas.symmetrise(cov_x, 'l')
    return mean_x, cov_x[0]


def estimate_rigid_transform_3d(pts1, pts2):
    """
    estimate_3d_transform(pts1, pts2) -> [R|t]
    Estimate 3D transformation using SVD.
    D.W. Eggert, A. Lorusso, R.B. Fisher, "Estimating 3-D rigid body 
    transformations: a comparison of four major algorithms", Machine Vision 
    and Applications (1997) 9: 272–290 Machine Vision and Applications, 
    Springer-Verlag 1997
    """
    assert ((pts1.shape == pts2.shape) and (pts1.ndim == 2) and
            (pts1.shape[1] == 3)), "pts1 and pts2 must be Nx3 ndarrays"
    # Compute the centroids
    centroid_p1 = pts1.mean(axis=0)
    centroid_p2 = pts2.mean(axis=0)
    # Subtract the centroid from the points
    pts1_centred = np.asanyarray(pts1 - centroid_p1, order='C')
    pts2_centred = np.asanyarray(pts2 - centroid_p2, order='C')
    # Compute the correlation
    correlation_matrix = blas.dger(pts1_centred, pts2_centred).sum(axis=0)
    U, S, Vt = np.linalg.svd(correlation_matrix)
    est_rot = np.dot(Vt.T, U.T)
    # Check if we estimated a reflection rather than a rotation
    if np.linalg.det(est_rot) < 0:
        Vt[-1] *= -1
        est_rot = np.dot(Vt.T, U.T)
    # Obtain the translation
    est_trans = centroid_p2 - np.dot(est_rot, centroid_p1[:, np.newaxis]).T
    return np.hstack((est_rot, est_trans.T))
    

###############################################################################
###############################################################################
#                                                                             #
# Copyright (C) 2010 Edward d'Auvergne                                        #
#                                                                             #
# This file is part of the program relax (http://www.nmr-relax.com).          #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################

# Module docstring.
"""Module for transforming between different coordinate systems."""

def cartesian_to_spherical(cart_vect, spherical_vect=None):
    """Convert the Cartesian vector [x; y; z] to spherical coordinates [r; theta; phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """
    
    if spherical_vect is None:
        spherical_vect = np.empty(cart_vect.shape)
    
    # The radial distance.
    #r = norm(vector)
    spherical_vect[0] = np.sqrt((cart_vect**2).sum(axis=0))
    r = spherical_vect[0]
    # Unit vector.
    unit = cart_vect / r

    # The polar angle.
    spherical_vect[1] = np.arccos(unit[2])

    # The azimuth.
    spherical_vect[2] = np.arctan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return spherical_vect


def spherical_to_cartesian(spherical_vect, cart_vect=None):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """
    
    if cart_vect is None:
        cart_vect = np.empty(spherical_vect.shape)
    # Trig alias.
    sin_theta = np.sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * np.cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * np.sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * np.cos(spherical_vect[1])
    
    return cart_vect
