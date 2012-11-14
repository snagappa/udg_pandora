# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:20:27 2012

@author: snagappa
"""
from image_geometry import cameramodels as ros_cameramodels
import numpy as np
import tf
import rospy
from lib.common import blas, misctools
from lib.common.misctools import STRUCT, rotation_matrix
import cv2
import image_feature_extractor
import copy
import code
from geometry_msgs.msg import PointStamped

default_float = "float64"

def _raise_(ex):
    raise ex

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

# transform listener
tflistener = None
TF_AVAILABLE = False

def transform_numpy_array(target_frame, source_frame, numpy_points,
                          timestamp=None, ROTATE_BEFORE_TRANSLATE=True):
    assert (((numpy_points.ndim == 2) and (numpy_points.shape[1] == 3)) or
            (np.prod(numpy_points.shape) == 0)), (
            "Points must be Nx3 numpy array")
    if numpy_points.shape[0] == 0:
        return np.empty(0)
    
    point = PointStamped()
    if not timestamp is None:
        point.header.stamp = timestamp
    #else:
    #    point.header.stamp = rospy.Time.now()
    point.header.frame_id = source_frame
    
    arr_len = numpy_points.shape[0]
    mat44 = tflistener.asMatrix(target_frame, point.header)
    
    if ROTATE_BEFORE_TRANSLATE:
        homogenous_points = np.hstack((numpy_points, np.ones((arr_len, 1))))
        tf_np_points = np.dot(mat44, homogenous_points.T)[:3].T
    else:
        translation = mat44[:3, 3]
        rot_mat = mat44[:3, :3]
        tf_np_points = np.dot(rot_mat, (numpy_points + translation).T).T
    return tf_np_points

#def transform_numpy_array(target_frame, source_frame, numpy_points,
#                          timestamp=None, ROTATE_BEFORE_TRANSLATE=True):
#    assert (((numpy_points.ndim == 2) and (numpy_points.shape[1] == 3)) or
#            (np.prod(numpy_points.shape) == 0)), (
#            "Points must be Nx3 numpy array")
#    if numpy_points.shape[0] == 0:
#        return np.empty(0)
#    
#    if timestamp is None:
#        timestamp = rospy.Time()
#    
#    trans, quaternion_rot = tflistener.lookupTransform(target_frame, 
#                                                       source_frame, timestamp)
#    rpy = np.asarray(tf.transformations.euler_from_quaternion(quaternion_rot))
#    rot_matrix = np.array(rotation_matrix(rpy), order='C')
#    trans = np.array(trans)
#    
#    if ROTATE_BEFORE_TRANSLATE:
#        tf_np_points = np.dot(rot_matrix, numpy_points.T).T + trans
#    else:
#        tf_np_points = np.dot(rot_matrix, (numpy_points + trans).T).T
#    return tf_np_points

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
    

class _FoV_(object):
    def __init__(self):
        """Class to model field of view"""
        # Probability of target detection
        self.pd = 0.98
        # Near distance of the field of view
        self.fov_near = 0.3
        # Far distance of the field of view
        self.fov_far = 5.0
        # Observation volume
        self.observation_volume = 0.0
        # tf frame variables
        self.tferror = None
        self.tfFrame = None
        # Helper for using the pointcloud interface
        self.pcl_helper = misctools.pcl_xyz(default_float)
        
        # Initialise transform listener
        global tflistener
        global TF_AVAILABLE
        if tflistener is None:
            try:
                tflistener = tf.TransformListener()
                TF_AVAILABLE = True
            except (rospy.ROSInitException, rospy.ROSException, 
                    rospy.ServiceException) as tferror:
                TF_AVAILABLE = False
                tferror = tferror
                tflistener = STRUCT()
                tflistener.transformPointCloud = (
                    lambda *args, **kwargs: _raise_(tferror))
                print tferror
                print "Error initialising tflistener, may be unavailable"
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=5.0):
        """set_near_far_fov(self, fov_near=0.3, fov_far=5.0) -> None
        Set the near and far distances visible by the camera
        """
        self.fov_near = fov_near
        self.fov_far = fov_far
        self.observation_volume = self._observation_volume_()
    
    def set_x_y_far(self, x_deg=None, y_deg=None, far_m=None):
        """Compatibility Function - sets the far distance only"""
        self.set_near_far_fov(self.fov_near, far_m)
    
    def fov_vertices_2d(self):
        """fov_vertices_2d(self) -> fov_vertices
        Generate a list of vertices describing the field of view in 2D
        Cartesian coordinates"""
        num_points = 10
        radius = np.ones(num_points)
        theta = np.linspace(0, 2*np.pi, num_points)
        phi = np.zeros(num_points)
        cart_coords = misctools.spherical_to_cartesian(
            np.asarray([radius, theta, phi]))
        cart_coords = cart_coords[[0, 2]].T
        return cart_coords
    
    def _observation_volume_(self):
        return 4/3*np.pi*(self.fov_far**3 - self.fov_near**3)
    
    def set_const_pd(self, pd=0.9):
        """set_const_pd(self, pd=0.9) -> None
        Set the probability of target detection for the model
        """
        self.pd = pd
    
    def pdf_detection(self, points1, points2=None, **kwargs):
        """pdf_detection(self, points) -> pd
        Returns the probability of detection for points between the spheres of
        radius fov_near and fov_far.
        """
        if not points1.shape[0]:
            return np.array([], dtype=np.bool)
        euclid_distance = np.power(np.power(points1, 2).sum(axis=1), 0.5)
        pd = np.zeros(points1.shape[0])
        pd[np.logical_and(self.fov_near < euclid_distance, 
                          euclid_distance < self.fov_far)] = self.pd
        return pd
    
    def is_visible(self, points1, points2=None, **kwargs):
        """is_visible(self, points) -> bool_vector
        """
        if not points1.shape[0]:
            return np.array([], dtype=np.bool)
        pd = self.pdf_detection(points1, points2, **kwargs)
        return (pd > 0)
    
    def pdf_clutter(self, points1, points2=None, **kwargs):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = self.pdf_detection(points1, points2, **kwargs).astype(np.bool)
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points1.shape[0])
        clutter_pdf[pd == False] = 1
        return 
    
    def _perform_tf_(self, target_frame, pcl_msg, RETURN_NP_ARRAY=False):
        print "CALLING _PERFORM_TF_"
        pcl_tf_points = tflistener.transformPointCloud(target_frame, 
                                                            pcl_msg)
        if RETURN_NP_ARRAY:
            return self.pcl_helper.from_pcl(pcl_tf_points)
        else:
            return pcl_tf_points
    
    def set_tf_frame(self, frame_id):
        self.tfFrame = frame_id
    
    #def _nparray_to_pcl_(self, numpy_points, frame_id):
    #    assert (((numpy_points.ndim == 2) and (numpy_points.shape[1] == 3)) or
    #        (np.prod(numpy_points.shape) == 0)), (
    #        "Points must be Nx3 numpy array")
    #    pcl_points = self.pcl_helper.to_pcl(numpy_points)
    #    pcl_points.header.stamp = rospy.Time.now()
    #    pcl_points.header.frame_id = frame_id
    #    return pcl_points
        
    def to_world_coords(self, numpy_points):
        """to_world_coords(self, numpy_points)->world_points
        Convert Nx3 numpy array of points from camera coordinate system to
        world coordinates"""
        #pcl_points = self._nparray_to_pcl_(numpy_points, self.tfFrame)
        #return self.to_world_coords_pcl(pcl_points, True)
        target_frame = "world"
        source_frame = self.tfFrame
        return transform_numpy_array(target_frame, source_frame, numpy_points,
                                     ROTATE_BEFORE_TRANSLATE=True)
    
    #def to_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False):
    #    """to_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False)
    #    -> world_points OR world_points_pcl
    #    Convert pointcloud from camera coordinate system to world
    #    coordinates.
    #    Set RETURN_NP_ARRAY to True to return a numpy array instead of the
    #    pointcloud"""
    #    target_frame = "world"
    #    return self._perform_tf_(target_frame, pcl_msg, RETURN_NP_ARRAY)
    
    def from_world_coords(self, numpy_points):
        """from_world_coords(self, numpy_points)->(camera_points,)
        Convert Nx3 numpy array of points from world coordinate system to
        camera coordinates"""
        #pcl_points = self._nparray_to_pcl_(numpy_points, "world")
        #return self.from_world_coords_pcl(pcl_points, True)
        target_frame = self.tfFrame
        source_frame = "world"
        return (transform_numpy_array(target_frame, source_frame, numpy_points,
                                     ROTATE_BEFORE_TRANSLATE=True),)
    
    #def from_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False):
    #    """from_world_coords_pcl(self, pcl, RETURN_NP_ARRAY=False)
    #    ->camera_points OR camera_pcl
    #    Convert points from world coordinate system to camera coordinates.
    #    Set RETURN_NP_ARRAY to True to return a numpy array instead of the
    #    pointcloud"""
    #    target_frame = self.tfFrame
    #    return self._perform_tf_(target_frame, pcl_msg, RETURN_NP_ARRAY)
    
    def relative(self, target_coord_NED, target_coord_RPY, world_points_NED):
        if not world_points_NED.shape[0]: return np.empty(0)
        relative_position = world_points_NED - target_coord_NED
        rot_matrix = np.array([rotation_matrix(-target_coord_RPY)], order='C')
        relative_position = blas.dgemv(rot_matrix, relative_position)
        return relative_position
    
    #def relative_rot_mat(self, RPY):
    #    return np.array([rotation_matrix(-RPY)])
    
    def absolute(self, source_coord_NED, source_coord_RPY, source_points_NED):
        if not source_points_NED.shape[0]: return np.empty(0)
        rot_matrix = np.array([rotation_matrix(source_coord_RPY)], order='C')
        absolute_position = (blas.dgemv(rot_matrix, source_points_NED) + 
                             source_coord_NED)
        return absolute_position
    
    #def absolute_rot_mat(self, RPY):
    #    return np.array([rotation_matrix(RPY)])


class DummyCamera(_FoV_):
    def __init__(self,  fov_x_deg=64, fov_y_deg=50, fov_far_m=3):
        _FoV_.__init__(self)
        self.fov_x_deg = fov_x_deg
        self.fov_y_deg = fov_y_deg
        self.fov_far_m = fov_far_m
        self.tmp = lambda: 0
        self.precalc()
    
    def set_x_y_far(self, x_deg=None, y_deg=None, far_m=None):
        if not x_deg == None:
            self.fov_x_deg = x_deg
        if not y_deg == None:
            self.fov_y_deg = y_deg
        if not far_m == None:
            self.fov_far_m = far_m
        self.precalc()
        
    def precalc(self):
        self.tmp.fov_x_rad = self.fov_x_deg * np.pi/180.0
        self.tmp.fov_y_rad = self.fov_y_deg * np.pi/180.0
        # Take cosine of half the angle
        self.tmp.tan_x = np.tan(self.tmp.fov_x_rad/2)
        self.tmp.tan_y = np.tan(self.tmp.fov_y_rad/2)
        self.tmp._test_dists_ = np.arange(0.05, self.fov_far_m, 0.05)
        self.tmp._test_dists_area_ = 4*self.get_rect__half_width_height(self.tmp._test_dists_).prod(axis=1)
        self.tmp._proportional_area_ = self.tmp._test_dists_area_/self.tmp._test_dists_area_.sum()
        self.tmp._proportional_vol_ = self.tmp._test_dists_area_*0.05/(0.33*self.tmp._test_dists_area_[-1]*self.fov_far_m)
    
    def is_visible(self, points1_xyz, points2_xyz=None, **kwargs):
        if not points1_xyz.shape[0]:
            return np.array([], dtype=np.bool)
        test_distances = points1_xyz[:, 0].copy()
        xy_limits = self.get_rect__half_width_height(test_distances)
        is_inside_rect = self.__inside_rect__(xy_limits, points1_xyz[:,1:3])
        return (is_inside_rect*((0 < test_distances)*np.logical_and(1.0<test_distances, test_distances<self.fov_far_m)))
        
    def __inside_rect__(self, half_rect__width_height, xy):
        bool_is_inside = ((-half_rect__width_height<xy)*(xy<half_rect__width_height)).all(axis=1)
        return bool_is_inside
        
    def fov_vertices_2d(self):
        x_delta = self.fov_far_m*self.tmp.tan_x
        return np.array([[0, 0], [-x_delta, self.fov_far_m], [x_delta, self.fov_far_m]])
        
    def get_rect__half_width_height(self, far):
        return np.array([far*self.tmp.tan_x, far*self.tmp.tan_y]).T
        
    def z_prob(self, far):
        #z_idx = abs(self.tmp._test_dists_[:, np.newaxis] - far).argmin(axis=0)
        #return self.tmp._proportional_vol_[z_idx].copy()
        
        return (1/(self.tmp._test_dists_area_[-1]*self.fov_far_m/3))*np.ones(far.shape[0])
        
    def pdf_detection(self, points1, points2=None, **kwargs):
        if not points1.shape[0]:
            return np.empty(0)
        # Transform points to local frame
        visible_features_idx = np.array(self.is_visible(points1), 
                                        dtype=np.float)
        # Take the distance rather than the square?
        dist_from_ref = np.sum(points1[:,[0, 1]]**2, axis=1)
        return np.exp(-dist_from_ref*0.005)*0.99*visible_features_idx
        
    def pdf_clutter(self, points1, points2=None, **kwargs):
        assert (((points1.ndim == 2) and (points1.shape[1] == 3)) or
                (np.prod(points1.shape) == 0)), "points1 must be a Nx3 ndarray"
        return self.z_prob(points1[:,0])
    

class PinholeCameraModel(ros_cameramodels.PinholeCameraModel, _FoV_):
    def __init__(self):
        """
        PinholeCameraModel() -> pinholecamera
        Pinhole camera model derived from image_geometry.PinholeCameraModel
        Additional methods to estimate detection probability and clutter pdf.
        """
        ros_cameramodels.PinholeCameraModel.__init__(self)
        self.tfFrame = None
        _FoV_.__init__(self)
        self._normals_ = np.empty(0)
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=5.0):
        """set_near_far_fov(self, fov_near=0.3, fov_far=5.0)
        Set near and far field of view
        """
        _FoV_.set_near_far_fov(self, fov_near, fov_far)
        self._create_normals_()
    
    def fromCameraInfo(self, msg):
        """fromCameraInfo(msg) -> None
        Create camera model using the camera info message
        """
        super(PinholeCameraModel, self).fromCameraInfo(msg)
        self.tfFrame = msg.header.frame_id
        self.observation_volume = self._observation_volume_()
        self._create_normals_()
    
    def project3dToPixel(self, point):
        """
        :param point:     numpy array of size Nx3 (x, y, z)->(right, up, far)
        
        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`projectPixelTo3dRay`.
        """
        src = np.empty((point.shape[0], 4), dtype=np.float64)
        src[:, :3] = point
        src[:, 3] = 1.0
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        w[w == 0] = np.inf
        pixels /= w[:, np.newaxis]
        return pixels
    
    def projectPixelTo3dRay(self, uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)
    
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        points = np.empty((uv.shape[0], 3))
        points[:, 0] = (uv[:, 0] - self.cx()) / self.fx()
        points[:, 1] = (uv[:, 1] - self.cy()) / self.fy()
        points[:, 2] = 1
        norm = np.sqrt((points**2).sum(axis=1))
        points /= norm[:, np.newaxis]
        return points
    
    def camera_matrix(self):
        return np.asarray(self.K)
    
    def projection_matrix(self):
        return np.asarray(self.P)
    
    def width(self):
        return self.width
    
    def height(self):
        return self.height
    
    def fov_vertices_2d(self):
        if not self.width is None:
            half_fov_angle = np.arctan2(self.width/2.0, self.fx())
            x_unit = np.tan(half_fov_angle)
            x_near = self.fov_near*x_unit
            x_far = self.fov_far*x_unit
            y_near = self.fov_near
            y_far = self.fov_far
            return np.asarray([[-x_near, y_near], [-x_far, y_far], 
                               [x_far, y_far], [x_near, y_near]])
        else:
            return np.empty((0, 2))
    
    def _observation_volume_(self):
        """_observation_volume_(self) -> obs_volume
        Calculate the volume of the observation space
        """
        # Diagonal corners of the pyramid base 
        corners_2d = np.asarray([(0, 0), (self.width, self.height)])
        unit_corners_3d = self.projectPixelTo3dRay(corners_2d)
        # Scale according to the z coordinate
        scalefactor = np.abs(self.fov_far/unit_corners_3d[0][2])
        corners_3d = unit_corners_3d*scalefactor
        base_width, base_height, _zero_ = corners_3d[1] - corners_3d[0]
        obs_volume = 0.33*base_width*base_height*self.fov_far
        return obs_volume
    
    def pdf_detection(self, points1, points2=None, **kwargs):
        """pdf_detection(self, points1, <margin=1e-2>) -> pd
        Determines the probability of detection for points specified according
        to (north, east, down) coordinate system relative to the camera centre.
        """
        if points1.shape[0] == 0:
            return np.empty(0)
        margin = kwargs.get("margin", 1e-2)
        # Convert points from (n,e,d) to camera (right, up, far)
        idx_visible = np.arange(points1.shape[0])
        points1 = points1[:, [1, 2, 0]]
        pd = np.zeros(points1.shape[0], dtype=np.float)
        # Check near plane
        idx_visible = idx_visible[points1[idx_visible, 2] > self.fov_near]
        if idx_visible.shape[0] == 0: return pd
        # Check far plane
        idx_visible = idx_visible[points1[idx_visible, 2] < self.fov_far]
        if idx_visible.shape[0] == 0: return pd
        # Frustum planes
        # Scale pd according to distance from the plane
        scale_factor = self.pd*np.ones(idx_visible.shape[0], dtype=np.float)
        for _normal_ in self._normals_:
            dot_product = np.dot(points1[idx_visible], _normal_)
            valid_idx = dot_product < margin
            idx_visible = idx_visible[valid_idx]
            scale_factor = scale_factor[valid_idx]
            dot_product = np.abs(dot_product[valid_idx])
            scale_idx = dot_product < 0.2
            scale_factor[scale_idx] *= dot_product[scale_idx]/0.2
            if idx_visible.shape[0] == 0: return pd
        pd[idx_visible] = scale_factor
        #pd[points[:, 2] < self.fov_near] = 0
        #pd[points[:, 2] > self.fov_far] = 0
        #pixels = self.project3dToPixel(points)
        #pd[pixels[:, 0] < px_margin] = 0
        #pd[pixels[:, 0] > (self.width+px_margin)] = 0
        #pd[pixels[:, 1] < px_margin] = 0
        #pd[pixels[:, 1] > (self.height+px_margin)] = 0
        return pd
    
    def _create_normals_(self):
        # Create unit vectors along edges of the view frustum
        edge_points = np.asarray([(0, 0), (self.width, 0), 
                                  (self.width, self.height), (0, self.height)])
        edge_unit_vectors = self.projectPixelTo3dRay(edge_points)
        #scalefactor = np.abs(self.fov_far/edge_unit_vectors[0, 2])
        #far_vectors = edge_unit_vectors * scalefactor
        # Create normals to the view frustum planes
        self._normals_ = np.asarray(
            [np.cross(edge_unit_vectors[idx], edge_unit_vectors[idx-1]) 
            for idx in range(edge_unit_vectors.shape[0])])
    
    def pdf_clutter(self, points1, points2=None, **kwargs):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = self.pdf_detection(points1, points2, **kwargs).astype(np.bool)
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points1.shape[0])
        clutter_pdf[pd == False] = 1
        return clutter_pdf
    
    def set_tf_frame(self, frame_id):
        self.tfFrame = frame_id
        self.tf_frame = frame_id
    

class StereoCameraModel(ros_cameramodels.StereoCameraModel, _FoV_):
    def __init__(self):
        """Idealised stereo camera"""
        self.left = PinholeCameraModel()
        self.right = PinholeCameraModel()
        _FoV_.__init__(self)
        self.tfFrame = None
    
    def fromCameraInfo(self, left_msg, right_msg):
        super(StereoCameraModel, self).fromCameraInfo(left_msg, right_msg)
        self.tfFrame = left_msg.header.frame_id
    
    def camera_matrix(self):
        return self.left.camera_matrix()
    
    def projection_matrix(self):
        return np.vstack((self.left.projection_matrix()[np.newaxis], 
                          self.right.projection_matrix()[np.newaxis]))
    
    def width(self):
        return self.left.width()
    
    def height(self):
        return self.left.height()
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=5.0):
        """set_near_far_fov(self, fov_near=0.3, fov_far=5.0) -> None
        Set the near and far field of view
        """
        self.left.set_near_far_fov(fov_near, fov_far)
        self.right.set_near_far_fov(fov_near, fov_far)
    
    def set_const_pd(self, pd=0.9):
        """set_const_pd(self, pd=0.9)
        Set a constant probability of detection for the targets in the scene
        """
        self.left.set_const_pd(pd)
        self.right.set_const_pd(pd)
    
    def fov_vertices_2d(self):
        if not self.left is None:
            return self.left.fov_vertices_2d()
        else:
            return np.empty((0, 2))
    
    def pdf_detection(self, points1, points2=None, **kwargs):
        """pdf_detection(self, points) -> pd
        Calculate the probability of detection for the points
        """
        l_pdf = self.left.pdf_detection(points1, None, **kwargs)
        if not points2 is None:
            r_pdf = self.right.pdf_detection(points2, None, **kwargs)
            l_pdf = np.vstack((l_pdf, r_pdf))
            return np.min(l_pdf, axis=0)
        else:
            return l_pdf
    
    def pdf_clutter(self, points1, points2=None, **kwargs):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the pdf of the clutter at the given points
        """
        clutter_l = self.left.pdf_clutter(points1, None, **kwargs)
        if not points2 is None:
            clutter_r = self.right.pdf_clutter(points2, None, **kwargs)
            replace_idx = clutter_l < clutter_r
            clutter_l[replace_idx] = clutter_r[replace_idx]
        return clutter_l
    
    def triangulate(self, pts_left, pts_right):
        """triangulate(self, pts_left, pts_right) -> points3d
        Triangulate points using rectified camera. pts_left and pts_right must
        be Nx2 numpy arrays
        """
        # Triangulate the points
        assert (pts_left.ndim == 2) and (pts_right.ndim == 2), "pts must be Nx2 numpy array"
        assert pts_left.shape == pts_right.shape, "pts_{left/right} must have the same shape"
        if pts_left.shape[0] == 0:
            return np.empty(0)
        points4d = cv2.triangulatePoints(np.asarray(self.left.P),
                                         np.asarray(self.right.P),
                                         pts_left.T, pts_right.T)
        points4d /= points4d[3]  #(wx,wy,wz,w) -> (x,y,z,1)
        points3d = np.asarray((points4d[0:3, :]).T, dtype=np.float, order='C')
        return points3d
    
    def from_world_coords(self, numpy_points):
        """from_world_coords(self, numpy_points)
            -> (camera_points_l, camera_points_r)
        Convert Nx3 numpy array of points from world coordinate system to
        camera coordinates"""
        return (self.left.from_world_coords(numpy_points)[0],
                self.right.from_world_coords(numpy_points)[0])
    
    def set_tf_frame(self, left_frame_id, right_frame_id):
        self.tfFrame = left_frame_id
        self.left.set_tf_frame(left_frame_id)
        self.right.set_tf_frame(right_frame_id)
    

class _CameraFeatureDetector_(object):
    def __init__(self, feature_extractor=image_feature_extractor.Orb, **kwargs):
        self._flann_mathcer_ = None
        self._featuredetector_ = None
        if not feature_extractor is None:
            self.set_feature_extractor(feature_extractor, **kwargs)
#        left = STRUCT()
#        left.raw = None
#        left.keypoints = None
#        left.descriptors = None
#        self.images = [left]
    
    def set_feature_extractor(self, 
                              feature_extractor=image_feature_extractor.Orb, **kwargs):
        self._featuredetector_ = feature_extractor(**kwargs)
        self._flann_matcher_ = FlannMatcher(self._featuredetector_.DESCRIPTOR_IS_BINARY)
    
    def get_features(self, image):
        (keypoints, descriptors) = self._featuredetector_.get_features(image)
        return keypoints, descriptors
    
    def detect_and_match(self, obj_kp, obj_desc, scene_kp, scene_desc, 
                           ratio=0.6):
        """
        _detect_and_match_(obj_kp, obj_desc, scene_kp, scene_desc, ratio)
        Returns pt1, pt2, valid_idx1, valid_idx2
        """
        if (obj_kp is None or (len(obj_kp) == 0) or 
            scene_kp is None or (len(scene_kp) == 0)):
            return (np.empty(0, dtype=np.float), np.empty(0, dtype=np.float),
                    np.empty(0, dtype=np.int), np.empty(0, dtype=np.int))
        idx1, idx2, distance = self._flann_matcher_.knnMatch(obj_desc, 
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
    
    def find_homography(self, pts_1, pts_2, method=cv2.RANSAC, 
                                       ransacReprojThreshold=5.0,
                                       min_inliers=10):
        """find_homography(self, pts_1, pts_2, method=cv2.RANSAC,
        ransacReprojThreshold=5.0, min_inliers=10) 
           -> status, h_mat, num_inliers, inliers_status
        Compute the homography from two matched sets of points
        """
        status = False
        h_mat = None
        num_inliers = 0
        inliers_status = np.empty(0, dtype=np.int)
        
        if pts_1.shape[0] > min_inliers:
            h_mat, inliers_status = cv2.findHomography(pts_1, pts_2, method, 
                                                       ransacReprojThreshold)
            inliers_status = np.squeeze(inliers_status)
            num_inliers = np.sum(inliers_status)
            if (num_inliers < min_inliers) or (h_mat is None):
                status = False
            else:
                status = True
        return status, h_mat, num_inliers, inliers_status
    

class PinholeCameraFeatureDetector(PinholeCameraModel, _CameraFeatureDetector_):
    def __init__(self, feature_extractor=image_feature_extractor.Orb, **kwargs):
        PinholeCameraModel.__init__(self)
        _CameraFeatureDetector_.__init__(self, feature_extractor, **kwargs)
    

class StereoCameraFeatureDetector(StereoCameraModel, _CameraFeatureDetector_):
    def __init__(self, feature_extractor=image_feature_extractor.Orb, **kwargs):
        """StereoCameraFeatureDetector(feature_extractor=image_feature_extractor.Orb)
        -> stereocam_detector
        """
        StereoCameraModel.__init__(self)
        _CameraFeatureDetector_.__init__(self, feature_extractor, **kwargs)
#        self.images.append(copy.deepcopy(self.images[0]))
    
    def points3d_from_img(self, image_left, image_right, ratio_threshold=0.75):
        """get_points(self, image_left, image_right, ratio_threshold=0.6)
            -> points3d, keypoints, descriptors
        """
        images = [STRUCT(), STRUCT()]
        # Get keypoints and descriptors from images
        for (idx, _im_) in zip((0, 1), (image_left, image_right)):
            #self.images[idx].raw = _im_.copy()
            (images[idx].keypoints, images[idx].descriptors) = (
            self.get_features(_im_))
        
        # Use the flann matcher to match keypoints
        im_left = images[0]
        im_right = images[1]
        if len(im_left.keypoints) and len(im_right.keypoints):
            match_result = self._flann_matcher_.detect_and_match(
                im_left.keypoints, im_left.descriptors,
                im_right.keypoints, im_right.descriptors, ratio_threshold)
            pts_l, pts_r, idx_l, idx_r, mask_tuple = match_result
        else:
            pts_l = np.empty(0)
            pts_r = np.empty(0)
        
        # Only proceed if there are matches
        if pts_l.shape[0]:
            kp_l = np.asarray(im_left.keypoints)[idx_l]
            desc_l = np.asarray(im_left.descriptors)[idx_l]
            kp_r = np.asarray(im_right.keypoints)[idx_r]
            desc_r = np.asarray(im_right.descriptors)[idx_r]
            # Valid matches are those where y co-ordinate of p1 and p2 are
            # almost equal
            y_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
            valid_disparity_mask = y_diff < 4.0
            # Select keypoints and descriptors which satisfy disparity
            pts_l = pts_l[valid_disparity_mask]
            kp_l = kp_l[valid_disparity_mask]
            desc_l = desc_l[valid_disparity_mask]
            pts_r = pts_r[valid_disparity_mask]
            kp_r = kp_r[valid_disparity_mask]
            desc_r = desc_r[valid_disparity_mask]
            
            # Triangulate the points now that they are matched
            # Normalise the points
            points3d = self.triangulate(pts_l, pts_r)
        else:
            points3d = np.empty(0)
            kp_l = np.empty(0)
            kp_r = np.empty(0)
            desc_l = np.empty(0)
            desc_r = np.empty(0)
        return points3d, (pts_l, pts_r), (kp_l, kp_r), (desc_l, desc_r)
    
