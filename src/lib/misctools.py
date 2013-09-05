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

import roslib
roslib.load_manifest('udg_pandora')

import blas
import numpy as np
from scipy import weave
from sensor_msgs.msg import PointCloud2, PointField
import pc2wrapper
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import scipy
import scipy.ndimage
import rospy
import threading
import message_filters
import copy
from sensor_msgs.msg import CameraInfo, Image
import pickle
#from operator import mul, add
#import code


_2_pi_ = 2*np.pi
_log_2_pi_ = np.log(_2_pi_)


class STRUCT(object):
    def __init__(self):
        self.__objvars__ = dir(self)
        self.__objvars__ += ["__objvars__", "__dtype__"]
    def __repr__(self):
        lines = []
        allvars = dir(self)
        [lines.append(
            _var_ + " = " + str(getattr(self, _var_)) + 
            self.__dtype__(getattr(self, _var_)))
            for _var_ in allvars if not _var_ in self.__objvars__]
        return "\n".join(lines)
    def __dtype__(self, var):
        dtype = "\nType: " + str(type(var))
        if type(var) is np.ndarray:
            dtype += ", " + str(var.dtype) + ", " + str(var.shape)
        return dtype
    

class message_buffer(object):
    def __init__(self, message_topic_list, message_type_list):
        assert not isinstance(message_topic_list, basestring)
        assert not isinstance(message_type_list, basestring)
        self._topics_ = copy.deepcopy(message_topic_list)
        self._topic_types_ = copy.deepcopy(message_type_list)
        
        # Storage for messages
        self._messages_ = [None for _topic_ in message_topic_list]
        self._message_time_ = rospy.Time(0)
        
        # List of callbacks and rates
        self._callbacks_ = {}
        self._last_callback_id_ = -1
        self._last_callback_time_ = []
        self._lock_ = threading.Lock()

        # Subscribe to the topics
        # Subscriber for the topics
        self._sub_ = None
        self._timesync_ = None
        self._subscribe_()
    
    def _subscribe_(self):
        self._sub_ = [
            message_filters.Subscriber(_topic_, _type_)
            for (_topic_, _type_) in zip(self._topics_, self._topic_types_)]
        self._timesync_ = message_filters.TimeSynchronizer(self._sub_, 10)
        self._timesync_.registerCallback(self._update_messages_)
    
    def _update_messages_(self, *args):
        self._lock_.acquire()
        if len(self._callbacks_):
            self._messages_ = [_msg_ for _msg_ in args]
            self._message_time_ = self._messages_[0].header.stamp
            self._start_callback_threads_()
        self._lock_.release()

    def register_callback(self, callback, rate=None):
        self._lock_.acquire()
        self._last_callback_id_ += 1
        time_delay = None if rate is None else rospy.Duration(1./rate)
        self._callbacks_[self._last_callback_id_] = {"callback":callback,
                                                     "timedelay":time_delay,
                                                     "lasttime":rospy.Time(0),
                                                     "block":0}
        self._lock_.release()
        return self._last_callback_id_

    def unregister_callback(self, callback_id):
        self._lock_.acquire()
        try:
            self._callbacks_.pop(callback_id)
        except KeyError:
            print "callback_id not found. already unregistered?"
        finally:
            self._lock_.release()

    def _start_callback_threads_(self):
        #self._lock_.acquire()
        threads = []
        unregister_callbacks_list = []
        for _callback_ in self._callbacks_:
            try:
                callback_info = self._callbacks_[_callback_]
                if ((not callback_info["block"]) and
                    (callback_info["timedelay"] is None or
                    self._message_time_ - callback_info["lasttime"] >= callback_info["timedelay"])):
                    threads.append(threading.Thread(target=self._thread_wrapper_, args=(callback_info,)))
                    threads[-1].start()
                    callback_info["lasttime"] = copy.copy(self._message_time_)
            except:
                print "error in callback, unregistering..."
                unregister_callbacks_list.append(_callback_)
        for _callback_ in unregister_callbacks_list:
            self._callbacks_.pop(_callback_)
        #self._lock_.release()

    def _thread_wrapper_(self, callback_info):
        callback_info["block"] = 1
        callback_info["callback"](*self._messages_)
        callback_info["block"] = 0

class camera_buffer(message_buffer):
    def __init__(self, camera_root, image_topic, IS_STEREO=False):
        assert type(camera_root) == str, "camera_root must be a string"
        if len(camera_root):
            camera_root = camera_root.rstrip('/') + '/'
        image_topic = image_topic.lstrip('/')
        if IS_STEREO:
            message_topics = [camera_root+"left/"+image_topic,
                             camera_root+"right/"+image_topic]
            message_types = [Image, Image]
        else:
            message_topics = [camera_root+image_topic]
            message_types = [Image]
        message_buffer.__init__(self, message_topics, message_types)
        self._is_stereo_ = IS_STEREO
        
        if IS_STEREO:
            # Initialise topics as None
            self._camera_info_topics_ = [camera_root+"left/camera_info",
                                        camera_root+"right/camera_info"]
            self._camera_info_ = (None, None)
        else:
            self._camera_info_topics_ = [camera_root+"camera_info"]
            self._camera_info_ = (None,)
        # Try to subscribe to the camera info
        self._cam_info_from_topics_()
    
    def _cam_info_from_topics_(self):
        try:
            self._camera_info_ = tuple(
                [rospy.wait_for_message(cam_info_topic, CameraInfo, 2)
                 for cam_info_topic in self._camera_info_topics_])
        except rospy.ROSException:
            rospy.logerr("Could not read camera parameters")
            camera_pickle_file = "bumblebee_new.p"
            print "Loading information from "+camera_pickle_file
            camera_info_pickle = (roslib.packages.get_pkg_dir("udg_pandora")+
            "/src/lib/" + camera_pickle_file)
            try:
                self._camera_info_ = tuple(
                    pickle.load(open(camera_info_pickle, "rb")))
            except IOError:
                print "Failed to load camera information!"
                rospy.logerror("Could not read camera parameters")
                raise rospy.exceptions.ROSException(
                    "Could not read camera parameters")
    
    def fromCameraInfo(self, camera_info_left, camera_info_right=None):
        if self._is_stereo_:
            self._camera_info_ = tuple([copy.deepcopy(_info_)
                for _info_ in (camera_info_left, camera_info_right)])
        else:
            self._camera_info_ = (copy.deepcopy(camera_info_left),)
    
    def get_camera_info(self, idx=None):
        if not idx is None:
            return (self._camera_info_[idx],)
        else:
            return self._camera_info_


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

class FlannMatcher(object):
    """
    Wrapper class for using the Flann matcher. Attempts to use the new 
    FlannBasedMatcher interface, but uses the fallback flann_Index if this is
    unavailable.
    """
    def __init__(self, DESCRIPTOR_IS_BINARY=False):
        if DESCRIPTOR_IS_BINARY:
            self.PARAMS = dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        else:
            self.PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, 
                                       trees = 5)
        self.NEW_FLANN_MATCHER = False
        try:
            # Use the cv2 FlannBasedMatcher if available
            # bug : need to pass empty dict (#1329)        
            self._flann_ = cv2.FlannBasedMatcher(self.PARAMS, {})
            self.NEW_FLANN_MATCHER = True
        except AttributeError as attr_err:
            print attr_err
            print "Could not initialise FlannBasedMatcher, using fallback"
    
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
            #try:
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
            #except as exc_err:
            #    print "error occurred while matching descriptors"
            #    code.interact(local=locals())
        else:
            self._flann_ = cv2.flann_Index(trainDescriptors, self.PARAMS)
            # Perform nearest neighbours search
            # bug: need to provide empty dict for params
            idx2, distance = self._flann_.knnSearch(queryDescriptors, k, 
                                                    params={})
        idx1 = np.arange(len(queryDescriptors))
        return idx1, idx2, distance
    
    def detect_and_match(self, obj_kp, obj_desc, scene_kp, scene_desc, ratio):
        """
        detect_and_match(self, obj_kp, obj_desc, scene_kp, scene_desc, ratio)
        Returns pt1, pt2, valid_idx1, valid_idx2
        """
        try:
            idx1, idx2, distance = self.knnMatch(obj_desc, scene_desc, 2)
        except:
            print "Error occurred computing knnMatch"
            print sys.exc_info()
            idx1 = np.empty(0, dtype=np.int)
            idx2 = np.empty((0, 2), dtype=np.int)
            distance = np.zeros((0, 2))
        
        # Use only good matches
        mask = distance[:, 0] < (distance[:, 1] * ratio)
        mask[idx2[:, 1] == -1] = False
        valid_idx1 = idx1[mask]
        idx2 = idx2[:, 0]
        valid_idx2 = idx2[mask]
        match_kp1, match_kp2 = [], []
        for (_idx1_, _idx2_) in zip(valid_idx1, valid_idx2):
            match_kp1.append(obj_kp[_idx1_])
            match_kp2.append(scene_kp[_idx2_])
        pts_1 = np.asarray([kp.pt for kp in match_kp1], dtype=np.float32)
        pts_2 = np.asarray([kp.pt for kp in match_kp2], dtype=np.float32)
        #kp_pairs = zip(match_kp1, match_kp2)
        return pts_1, pts_2, valid_idx1, valid_idx2, (idx1, idx2, mask) #, kp_pairs
    

class image_converter:
    def __init__(self):
        """image_converter() -> converter
        Create object to convert sensor_msgs.msg.Image type to cv_image
        """
        self.bridge = CvBridge()
        self._cv_image_ = None;
    
    def cvimage(self, data, COLOUR_FMT=None): #or set format to 'passthrough'?
        """cvimage(self, data, COLOUR_FMT=None) -> cv_image
        Convert data from ROS sensor_msgs Image type to OpenCV using the 
        specified colour format
        """
        try:
            if COLOUR_FMT is None:
                cv_image = self.bridge.imgmsg_to_cv(data)
            else:
                cv_image = self.bridge.imgmsg_to_cv(data, COLOUR_FMT)
        except CvBridgeError, e:
            self._cv_image_ = None
            print e
        self._cv_image_ = cv_image
        return self._cv_image_
        
    def cvimagegray(self, data):
        """cvimage(self, data, COLOUR_FMT=None) -> cv_image
        Convert data from ROS sensor_msgs Image type to OpenCV with colour
        format "mono8"
        """
        return self.cvimage(data, 'mono8') # Use 'rgb8' instead?
    
    def img_msg(self, cvim, encoding="passthrough"):
        """img_msg(self, cvim, encoding="passthrough") -> imgmsg
        Convert OpenCV image to ROS sensor_msgs Image
        """
        return self.bridge.cv_to_imgmsg(cvim, encoding)
    

class pcl_msg_helper(object):
    def __init__(self, msg_fields, datatype_str="float32"):
        self.datatype = getattr(np, datatype_str)
        self.pointfield_dtype = getattr(PointField, datatype_str.upper())
        self.data_size = self.datatype(0).nbytes
        self.msg_fields = msg_fields
        num_fields = len(self.msg_fields)
        field_offsets = range(0, num_fields*self.data_size, self.data_size)
        
        self.pcl_fields = [PointField(name=_field_name_, offset=_field_offset_,
                                      datatype=self.pointfield_dtype, count=1)
                           for (_field_name_, _field_offset_) in 
                           zip(self.msg_fields, field_offsets)]
        #self.pcl_header = PointCloud2().header
    
    def to_pcl(self, point_array):
        point_array = np.asarray(point_array, dtype=self.datatype)
        return pc2wrapper.create_cloud(PointCloud2().header, self.pcl_fields, 
                                       point_array)
    
    def from_pcl(self, pcl_msg):
        if self.msg_fields is None:
            pcl_points = np.array(list(pc2wrapper.read_points(pcl_msg)))
        else:
            pcl_fields_list = [pcl_msg.fields[i].name 
                               for i in range(len(pcl_msg.fields))]
            if not pcl_fields_list == self.msg_fields:
                print "pcl fields don't match expected values; ignoring..."
                pcl_points = np.empty(0)
            else:
                pcl_points = np.array(list(pc2wrapper.read_points(pcl_msg)))
        return(pcl_points)
    

def pcl_xyz(datatype="float32"):
    fieldnames = ['x', 'y', 'z']
    return pcl_msg_helper(fieldnames, datatype)
    
def pcl_xyz_cov(datatype="float32"):
    fieldnames = ['x', 'y', 'z'] + ['sigma_x', 'sigma_y', 'sigma_z']
    return pcl_msg_helper(fieldnames, datatype)

def rotation_matrix(RPY):
    ## See http://en.wikipedia.org/wiki/Rotation_matrix
    r, p, y = 0, 1, 2
    c = np.cos(RPY)
    s = np.sin(RPY)
    rot_matrix = np.array(
        [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
         [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
         [-s[p], s[r]*c[p], c[r]*c[p] ]])
    return rot_matrix

def relative_rot_mat(RPY):
    return np.array([rotation_matrix(-RPY)])
    
def absolute_rot_mat(RPY):
    return np.array([rotation_matrix(RPY)])

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
    assert P.shape[0] == 1, "P must be a 1xNxN ndarray"
    assert P.shape[1] == P.shape[2], "P must be a square matrix"
    select_diag_idx = range(P.shape[1])
    residual = x-y
    #diag_P = P[:, select_diag_idx, select_diag_idx]
    diag_P = P[0, select_diag_idx, select_diag_idx]
    p_times_residual = (1.0/diag_P)*residual
    distance = (residual*p_times_residual).sum(1)**0.5
    #distance = (blas.ddot(residual, p_times_residual))**0.5
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
    #merged_P = np.array(
    #    [(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], 
    #     order='C')
    #blas.symmetrise(merged_P, 'l')
    
    # method 2:
    #P_copy = P + blas.dger(residual, residual)
    
    # method 3:
    P_copy = P + [np.dot(residual[np.newaxis,i].T, residual[np.newaxis,i])
        for i in range(residual.shape[0])]
    
    #merged_P = np.array(
    #    [(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], 
    #     order='C')
    merged_P = (wt[:, np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt
    return merged_wt, merged_x, merged_P
    
def get_resample_index(weights, nparticles=-1):
    weights = weights/weights.sum()
    
    if nparticles == -1:
        nparticles = weights.shape[0]
    
    resampled_indices = np.empty(nparticles, dtype=int)
    wt_cdf = np.empty(weights.shape, dtype=float)
    u1 = np.random.uniform()/nparticles
    
    python_vars = ['nparticles', 'weights', 'wt_cdf', 'u1', 
                   'resampled_indices']
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
    
    
def mvnpdf(x, mu, sigma, LOG=False):
    # Compute the residuals
    #if x.shape[0] == 1:
    #    residual = np.repeat(x, mu.shape[0], 0)
    #else:
    #    residual = x.copy(order='c')
    #blas.daxpy(-1.0, mu, residual)
    residual = np.asarray(x-mu, order='C')
    #if x.shape[0] == 1:
    #    x = np.repeat(x, mu.shape[0], 0)
    #residual = x-mu
    chol_sigma = blas.dpotrf(sigma)
    # Compute the determinant
    diag_range = range(chol_sigma.shape[1])
    diag_vec = chol_sigma[:, diag_range, diag_range]
    #diag_vec = np.array([np.diag(chol_sigma[i]) 
    #                    for i in range(chol_sigma.shape[0])])
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
    
    P = residual.shape[1]
    if LOG:
        pdf = -0.5*exp_term - 0.5*(np.log(det_sigma)+P*_log_2_pi_)
    else:
        pdf = np.exp(-0.5*exp_term)/np.sqrt(det_sigma*(_2_pi_)**P)
    return pdf
    
    
def approximate_mvnpdf(x, mu, sigma, LOG=False):
    residual = x-mu
    # Extract diagonals from sigma into a 2D array
    sigma_diag = sigma[:, range(sigma.shape[1]), range(sigma.shape[2])]
    exp_term = ((residual**2)*(1.0/sigma_diag)).sum(axis=1)
    P = residual.shape[1]
    if LOG:
        pdf = -0.5*exp_term - 0.5*(np.log(sigma_diag).sum(axis=1)+P*_log_2_pi_)
    else:
        pdf = np.exp(-0.5*exp_term)/((sigma_diag.prod(axis=1)*(_2_pi_)**P)**0.5)
    return pdf


def sample_mn_cv(x, wt=None, SYMMETRISE=False):
    if wt.shape[0] == 1:
        return x[0].copy(), np.zeros((x.shape[1], x.shape[1]))
    if wt == None:
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
    mean_x = (wt[:, np.newaxis]*x).sum(axis=0)
    residuals = x - mean_x
    cov_x = np.array(
        [blas.dsyr('l', residuals, wt).sum(axis=0)/(1-(wt**2).sum())])
    if SYMMETRISE:
        blas.symmetrise(cov_x, 'l')
    return mean_x, cov_x[0]


def circmean(samples, high=np.pi, low=-np.pi, axis=None, weights=None):
    """
Compute the weighted circular mean for samples in a range
(modified from scipy.stats.circmean).

Parameters
----------
samples : array_like
Input array.
high : float or int, optional
High boundary for circular mean range. Default is ``2*pi``.
low : float or int, optional
Low boundary for circular mean range. Default is 0.
axis : int, optional
Axis along which means are computed. The default is to compute
the mean of the flattened array.

Returns
-------
circmean : float
Circular mean.

"""
    ang = (samples - low)*2*np.pi / (high-low)
    res = np.angle(np.average(np.exp(1j*ang), axis=axis, weights=weights))
    mask = res < 0
    if (mask.ndim > 0):
        res[mask] += 2*np.pi
    elif mask:
        res = res + 2*np.pi
    return res*(high-low)/2.0/np.pi + low


def normalize_angle(np_array):
    return (np_array + (2.0*np.pi*np.floor((np.pi - np_array)/(2.0*np.pi))))


def estimate_rigid_transform_3d(pts1, pts2):
    """
    estimate_3d_transform(pts1, pts2) -> [R|t]
    Estimate 3D transformation from pts1 to pts2 using SVD.
    D.W. Eggert, A. Lorusso, R.B. Fisher, "Estimating 3-D rigid body 
    transformations: a comparison of four major algorithms", Machine Vision 
    and Applications (1997) 9: 272–290 Machine Vision and Applications, 
    Springer-Verlag 1997
    This code adapted from Nghia Ho
    http://nghiaho.com/uploads/code/rigid_transform_3D.py_
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
    """Convert the Cartesian vector [x; y; z] to spherical coordinates 
    [r; theta; phi].

    The parameter r is the radial distance, theta is the polar angle, and phi
    is the azimuth.


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
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian
    vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi
    is the azimuth.


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

# Part of the PsychoPy library 
# Copyright (C) 2010 Jonathan Peirce 
# Distributed under the terms of the GNU General Public License (GPL). 
def pol2cart(theta, radius, units='deg'):
    """Convert from polar to cartesian coordinates
    **usage**:
        x,y = pol2cart(theta, radius, units='deg')
    """
    if units in ['deg', 'degs']:
        theta = theta*np.pi/180.0 
    xx = radius*np.cos(theta) 
    yy = radius*np.sin(theta)
    return xx,yy

def cart2pol(x,y, units='deg'):
    """Convert from cartesian to polar coordinates
    **usage**:
        theta, radius = pol2cart(x, y, units='deg')
    units refers to the units (rad or deg) for theta that should be returned"""
    radius= np.hypot(x,y)
    theta= np.arctan2(y,x)
    if units in ['deg', 'degs']:
        theta=theta*180/np.pi
    return theta, radius 


# http://marc.info/?l=gimp-developer&m=118984615408912&w=4
#   Implementation of a retinex algorithm similar to that described by
#   John McCann in 1999

#   For more information about the algorithm see \
# http://www.cs.sfu.ca/~colour/publications/IST-2000/index.html #   Brian Funt, Florian Ciurea, and John \
# McCann "Retinex in Matlab," Proceedings of the IS&T/SID Eighth Color Imaging Conference: Color Science, \
# Systems and Applications, 2000, pp 112-121.

#   Copyright (C) 2007 John Fremlin
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.


class Retinex(object):
    def __init__(self):
        self.small_amount = 1/1024.0
        self.difference_from_neighbours_kernel = np.array([ [-1, -1, -1], [-1,8,-1], [-1,-1,-1]],"d")
        self.global_logscale = True
    
    def _shrink_(self, chan, scale):
        return scipy.ndimage.zoom(chan, 1/float(scale), prefilter=False, order=5)
    
    def _image_clip_(self, chan):
        if self.global_logscale:
            return chan.clip(-np.inf, 0.0)
        else:
            return chan.clip(0.0,1.0)
    
    def _retinex_at_scale_(self, retinex, orig, scale):
        assert(orig.size == retinex.size)
        working = orig
        diff = scipy.ndimage.convolve(working, self.difference_from_neighbours_kernel)
        result = (retinex + diff)/2
        working = (retinex + self._image_clip_(result))/2
        return working
    
    def _resize_(self, chan, new_size):
        orig = chan.shape
        zoom = [((new+0.9)/float(old)) for old, new in zip(orig, new_size)]
        ret = scipy.ndimage.zoom(chan, zoom, prefilter=False, order=5)
        assert(new_size == ret.shape)
        return ret

    def _process_one_channel_(self, chan):
        retinex = np.array([[chan.mean()]],"d")
        
        for logscale in range(int(np.log2(min(*chan.shape))), -1, -1):
            scale = 1 << logscale
            orig = self._shrink_(chan, scale)
            retinex = self._retinex_at_scale_(self._resize_(retinex, orig.shape), orig, scale)
        return retinex
    #    return np.abs(chan-retinex)

    def retinex(self, image, logscale=True):
        self.global_logscale = logscale
        if self.global_logscale:
            image = np.log(image+self.small_amount)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        [ width, height, channels ] = image.shape
        for c in range(channels):
            image[:,:,c] = self._process_one_channel_(image[:,:,c])
        
        if self.global_logscale:
            image = np.exp(image)-self.small_amount
        return image.squeeze()

