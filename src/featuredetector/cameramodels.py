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
from lib.common.misctools import STRUCT, FlannMatcher, rotation_matrix
import cv2
import image_feature_extractor
import copy
import code
from geometry_msgs.msg import PointStamped

default_float = "float64"

def _raise_(ex):
    raise ex

# transform listener
tflistener = None
TF_AVAILABLE = False

def transform_numpy_array(target_frame, source_frame, numpy_points,
                          timestamp=None):
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
    homogenous_points = np.hstack((numpy_points, np.ones((arr_len, 1))))
    tf_np_points = np.dot(mat44, homogenous_points.T)[:3].T
    return tf_np_points


class _FoV_(object):
    def __init__(self):
        """Class to model field of view"""
        # Probability of target detection
        self.pd = 0.9
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
    
    def pdf_detection(self, points, **kwargs):
        """pdf_detection(self, points) -> pd
        Returns the probability of detection for points between the spheres of
        radius fov_near and fov_far.
        """
        if not points.shape[0]:
            return np.array([], dtype=np.bool)
        euclid_distance = np.power(np.power(points, 2).sum(axis=1), 0.5)
        pd = np.zeros(points.shape[0])
        pd[np.logical_and(self.fov_near < euclid_distance, 
                          euclid_distance < self.fov_far)] = self.pd
        return pd
    
    def is_visible(self, points):
        """is_visible(self, points) -> bool_vector
        """
        if not points.shape[0]:
            return np.array([], dtype=np.bool)
        pd = self.pdf_detection(points)
        return (pd > 0)
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = bool(self.pdf_detection(points))
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points.shape[0])
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
                                     )#timestamp=rospy.Time.now())
    
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
        """from_world_coords(self, points)->camera_points
        Convert Nx3 numpy array of points from world coordinate system to
        camera coordinates"""
        #pcl_points = self._nparray_to_pcl_(numpy_points, "world")
        #return self.from_world_coords_pcl(pcl_points, True)
        target_frame = self.tfFrame
        source_frame = "world"
        return transform_numpy_array(target_frame, source_frame, numpy_points,
                                     )#timestamp=rospy.Time.now())
    
    #def from_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False):
    #    """from_world_coords_pcl(self, pcl, RETURN_NP_ARRAY=False)
    #    ->camera_points OR camera_pcl
    #    Convert points from world coordinate system to camera coordinates.
    #    Set RETURN_NP_ARRAY to True to return a numpy array instead of the
    #    pointcloud"""
    #    target_frame = self.tfFrame
    #    return self._perform_tf_(target_frame, pcl_msg, RETURN_NP_ARRAY)
    
    def relative(self, target_coord_NED, target_coord_PY, world_points_NED):
        if not world_points_NED.shape[0]: return np.empty(0)
        relative_position = world_points_NED - target_coord_NED
        rot_matrix = np.array([rotation_matrix(-target_coord_PY)])
        relative_position = blas.dgemv(rot_matrix, relative_position)
        return relative_position
    
    #def relative_rot_mat(self, RPY):
    #    return np.array([rotation_matrix(-RPY)])
    
    def absolute(self, source_coord_NED, source_coord_RPY, source_points_NED):
        if not source_points_NED.shape[0]: return np.empty(0)
        rot_matrix = np.array([rotation_matrix(source_coord_RPY)])
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
    
    def is_visible(self, point_xyz):
        if not point_xyz.shape[0]:
            return np.array([], dtype=np.bool)
        test_distances = point_xyz[:, 0].copy()
        xy_limits = self.get_rect__half_width_height(test_distances)
        is_inside_rect = self.__inside_rect__(xy_limits, point_xyz[:,1:3])
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
        
    def pdf_detection(self, points, **kwargs):
        if not points.shape[0]:
            return np.empty(0)
        # Transform points to local frame
        visible_features_idx = np.array(self.is_visible(points), 
                                        dtype=np.float)
        # Take the distance rather than the square?
        dist_from_ref = np.sum(points[:,[0, 1]]**2, axis=1)
        return np.exp(-dist_from_ref*0.005)*0.99*visible_features_idx
        
    def pdf_clutter(self, points):
        assert (((points.ndim == 2) and (points.ndim[1] == 3)) or
                (np.prod(points.shape) == 0)), "points must be a Nx3 ndarray"
        return self.z_prob(points[:,0])
    

class PinholeCameraModel(ros_cameramodels.PinholeCameraModel, _FoV_):
    def __init__(self):
        """
        Pinhole camera model derived from image_geometry.PinholeCameraModel
        Additional methods to estimate detection probability and clutter pdf.
        """
        ros_cameramodels.PinholeCameraModel.__init__(self)
        self.tfFrame = None
        _FoV_.__init__(self)
    
    def fromCameraInfo(self, msg):
        """fromCameraInfo(msg) -> None
        Create camera model using the camera info message
        """
        super(PinholeCameraModel, self).fromCameraInfo(msg)
        self.tfFrame = msg.header.frame_id
        self.observation_volume = self._observation_volume_()
    
    def project3dToPixel(self, point):
        """
        :param point:     numpy array of size Nx3 (x, y, z)->(right, up, far)
        
        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`projectPixelTo3dRay`.
        """
        src = np.empty((point.shape[0], 4), dtype=np.float)
        src[:, :3] = point
        src[:, 3] = 1.0
        P = np.asarray(self.P, dtype=np.float)
        dst = blas.dgemv(np.array([P], order='C'), src)
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        w[w == 0] = np.inf
        pixels /= w[:, np.newaxis]
        return pixels
    
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
        corners_2d = [(0, 0), (self.width, self.height)]
        unit_corners_3d = [np.asarray(self.projectPixelTo3dRay(_corner_))
                           for _corner_ in corners_2d]
        # Scale according to the z coordinate
        scalefactor = np.abs(self.fov_far/unit_corners_3d[0][2])
        corners_3d = unit_corners_3d*scalefactor
        base_width, base_height, _zero_ = corners_3d[1] - corners_3d[0]
        obs_volume = 0.33*base_width*base_height*self.fov_far
        return obs_volume
    
    def pdf_detection(self, points, **kwargs):
        """pdf_detection(self, points, {"px_margin":1e-2}) -> pd
        Determines the probability of detection for points specified according
        to (north, east, down) coordinate system relative to the camera centre.
        """
        if points.shape[0] == 0:
            return np.empty(0)
        px_margin = kwargs.get("px_margin", 1e-2)
        # Convert points from (n,e,d) to camera (right, up, far)
        points = points[:, [1, 2, 0]]
        pd = self.pd*np.ones(points.shape[0], dtype=np.float)
        pd[points[:, 2] < self.fov_near] = 0
        pd[points[:, 2] > self.fov_far] = 0
        pixels = self.project3dToPixel(points)
        try:
            pd[pixels[:, 0] < px_margin] = 0
            pd[pixels[:, 0] > (self.width+px_margin)] = 0
            pd[pixels[:, 1] < px_margin] = 0
            pd[pixels[:, 1] > (self.height+px_margin)] = 0
        except:
            code.interact(local=locals())
        return pd
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = self.pdf_detection(points).astype(np.bool)
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points.shape[0])
        clutter_pdf[pd == False] = 1
        return clutter_pdf
    

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
    
    def pdf_detection(self, points, **kwargs):
        """pdf_detection(self, points) -> pd
        Calculate the probability of detection for the points
        """
        l_pdf = self.left.pdf_detection(points, **kwargs)
        r_pdf = self.left.pdf_detection(points, **kwargs)
        return np.logical_and(l_pdf, r_pdf)
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the pdf of the clutter at the given points
        """
        return self.left.pdf_clutter(points)
    
    def triangulate(self, pts_left, pts_right):
        """triangulate(self, pts_left, pts_right) -> points3d
        Triangulate points using rectified camera. pts_left and pts_right must
        be Nx2 numpy arrays
        """
        # Triangulate the points
        points4d = cv2.triangulatePoints(np.asarray(self.left.P),
                                         np.asarray(self.right.P),
                                         pts_left.T, pts_right.T)
        points4d /= points4d[3]  #(wx,wy,wz,w) -> (x,y,z,1)
        points3d = np.asarray((points4d[0:3, :]).T, dtype=np.float, order='C')
        return points3d
    

class StereoCameraFeatureDetector(StereoCameraModel):
    def __init__(self, feature_extractor=image_feature_extractor.Orb):
        StereoCameraModel.__init__(self)
        self.featuredetector = feature_extractor()
        self._flann_ = FlannMatcher(self.featuredetector.DESCRIPTOR_IS_BINARY)
        self.images = STRUCT()
        self.images.left = STRUCT()
        self.images.right = STRUCT()
        self.images.left.raw = None
        self.images.left.keypoints = None
        self.images.left.descriptors = None
        self.images.right = copy.copy(self.images.left)
    
    def points3d_from_img(self, image_left, image_right,
                   ratio_threshold=0.75):
        """get_points(self, image_left, image_right, ratio_threshold=0.75)
            -> points3d, keypoints, descriptors
        """
        # Get keypoints and descriptors from images
        self.images.left.raw = image_left.copy()
        (self.images.left.keypoints, self.images.left.descriptors) = (
            self.featuredetector.get_features(image_left))
        self.images.right.raw = image_right.copy()
        (self.images.right.keypoints, self.images.right.descriptors) = (
            self.featuredetector.get_features(image_right))
        
        # Use the flann matcher to match keypoints
        im_left = self.images.left
        im_right = self.images.right
        match_result = self._flann_.matcher.detect_and_match(
            im_left.keypoints, im_left.descriptors,
            im_right.keypoints, im_right.descriptors, ratio_threshold)
        pts_l, pts_r, idx_l, idx_r, mask_tuple = match_result
        
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
        points3d = self.triangulate(pts_l, pts_r)
        return points3d, (kp_l, kp_r), (desc_l, desc_r)
    
