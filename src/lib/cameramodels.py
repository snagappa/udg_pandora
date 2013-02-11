# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:20:27 2012

@author: snagappa
"""
from image_geometry import cameramodels as ros_cameramodels
import numpy as np
import tf
import rospy
import blas
import misctools
import image_feature_extractor
from misctools import STRUCT, rotation_matrix, FlannMatcher
import cv2
import copy
import code
from geometry_msgs.msg import PointStamped
import sys
import traceback

default_float = "float64"

def _raise_(ex):
    raise ex

# transform listener
tflistener = None
TF_AVAILABLE = False


def get_transform(target_frame, source_frame, timestamp=None):
    """get_transform(target_frame, source_frame, timestamp=None)->mat44
    Get transformation matrix using tf
    """
    point = PointStamped()
    if not timestamp is None:
        point.header.stamp = timestamp
    point.header.frame_id = source_frame
    
    mat44 = tflistener.asMatrix(target_frame, point.header)
    return mat44

def transform_numpy_array(target_frame, source_frame, numpy_points,
                          timestamp=None, ROTATE_BEFORE_TRANSLATE=True):
    assert (((numpy_points.ndim == 2) and (numpy_points.shape[1] == 3)) or
            (np.prod(numpy_points.shape) == 0)), (
            "Points must be Nx3 numpy array")
    if numpy_points.shape[0] == 0:
        return np.empty(0)
    
    arr_len = numpy_points.shape[0]
    try:
        mat44 = get_transform(target_frame, source_frame, timestamp)
    except:
        sys_exc = sys.exc_info()
        print "Error converting from %s to %s" % (source_frame, target_frame)
        print "CAMERAMODELS:TRANSFORM_NUMPY_ARRAY():\n", traceback.print_tb(sys_exc[2])
        return np.empty(0, dtype=numpy_points.dtype)
    
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
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=3.0):
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
    
    def _pdf_detection_(self, rel_points1, rel_points2=None, **kwargs):
        """_pdf_detection_(self, rel_points1, rel_points2=None, **kwargs)->pd
        Returns the probability of detection for points lying between two
        spheres specified by fov_near and fov_far. Points are specified
        according to the camera coordinate system.
        """
        euclid_distance = np.power(np.power(rel_points1, 2).sum(axis=1), 0.5)
        pd = np.zeros(rel_points1.shape[0])
        pd[np.logical_and(self.fov_near < euclid_distance, 
                          euclid_distance < self.fov_far)] = self.pd
        return pd
    
    def pdf_detection(self, points, **kwargs):
        """pdf_detection(self, points) -> pd
        Returns the probability of detection for points specified according to 
        world co-ordinates.
        """
        if points.shape[0] == 0:
            return np.empty(0)
        rel_points = self.from_world_coords(points)[0]
        return self._pdf_detection_(rel_points, **kwargs)
    
    def is_visible(self, points, **kwargs):
        """is_visible(self, points) -> bool_vector
        """
        if points.shape[0] == 0:
            return np.empty(0, dtype=np.bool)
        pd = self.pdf_detection(points, **kwargs)
        return (pd > 0)
    
    def is_visible_relative2sensor(self, rel_points1, rel_points2=None, **kwargs):
        if rel_points1.shape[0] == 0:
            return np.empty(0, dtype=np.bool)
        pd = self._pdf_detection_(rel_points1, **kwargs)
        return (pd > 0)
    
    def pdf_clutter(self, points1, points2=None, **kwargs):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        #pd = self._pdf_detection_(points1, points2, **kwargs).astype(np.bool)
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points1.shape[0])
        #clutter_pdf[pd == 0] = 1
        return clutter_pdf
    
    def set_tf_frame(self, frame_id):
        self.tfFrame = frame_id
    
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
    
    def rotation_matrix(self):
        """rotation_matrix(self) -> rot_mat
        returns the 3x3 rotation matrix of the sensor with respect to the world
        """
        return get_transform(self.tfFrame, "world")[:3, :3]
    
    def inv_rotation_matrix(self):
        """rotation_matrix(self) -> rot_mat
        returns the 3x3 rotation matrix of the world with respect to the sensor
        """
        return get_transform("world", self.tfFrame)[:3, :3]
    
    def relative(self, target_coord_XYZ, target_coord_RPY, world_points_XYZ):
        if not world_points_XYZ.shape[0]: return np.empty(0)
        relative_position = world_points_XYZ - target_coord_XYZ
        rot_matrix = np.array([rotation_matrix(-target_coord_RPY)], order='C')
        relative_position = blas.dgemv(rot_matrix, relative_position)
        return relative_position
    
    def absolute(self, source_coord_XYZ, source_coord_RPY, source_points_XYZ):
        if not source_points_XYZ.shape[0]: return np.empty(0)
        rot_matrix = np.array([rotation_matrix(source_coord_RPY)], order='C')
        absolute_position = (blas.dgemv(rot_matrix, source_points_XYZ) + 
                             source_coord_XYZ)
        return absolute_position
    
    def observations(self, points):
        rel_states = self.from_world_coords(points)
        return rel_states
    
    def observation_jacobian(self):
        return self.inv_rotation_matrix()
    

class SphericalCamera(_FoV_):
    def __init__(self):
        _FoV_.__init__(self)
        # Observation volume
        self.observation_volume = self._observation_volume_()
    

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
        self._camera_info_ = None
    
    def copy(self):
        new_camera = PinholeCameraModel()
        if not self._camera_info_ is None:
            new_camera.fromCameraInfo(self._camera_info_)
        new_camera.set_tf_frame(self.tfFrame)
        return new_camera
    
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
        self._camera_info_ = msg
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
        if not point.shape[0]:
            return np.empty(0)
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
    
    def get_width(self):
        return self.width
    
    def get_height(self):
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
    
    def _pdf_detection_(self, rel_points1, rel_points2=None, **kwargs):
        """pdf_detection(self, points, <margin=1e-1>) -> pd
        Determines the probability of detection for points specified according
        to the camera coordinate system.
        """
        margin = kwargs.get("margin", 0)
        #USE_PROJECTION_METHOD = kwargs.get("USE_PROJECTION_METHOD", True)
        
        # Track indices of visible points
        idx_visible = np.arange(rel_points1.shape[0])
        # Initialise pd as 0
        pd = np.zeros(rel_points1.shape[0], dtype=np.float)
        # Check near plane
        idx_visible = idx_visible[rel_points1[idx_visible, 2] > (self.fov_near+margin)]
        if idx_visible.shape[0] == 0: return pd
        # Check far plane
        idx_visible = idx_visible[rel_points1[idx_visible, 2] < (self.fov_far-margin)]
        if idx_visible.shape[0] == 0: return pd
        
        # Frustum planes
        for _normal_ in self._normals_:
            dot_product = np.dot(rel_points1[idx_visible], _normal_)
            valid_idx = dot_product < 0
            idx_visible = idx_visible[valid_idx]
            if idx_visible.shape[0] == 0: return pd
        
        # Use projection or frustums to evaluate pd
        # Apply projection to points and discard points on the boundaries
        image_points = self.project3dToPixel(rel_points1[idx_visible])
        xmargin_l = 64
        xmargin_r = 64
        ymargin_t = 64
        ymargin_b = 64
        # Discard points with image x-coordinate < margin
        _valid_ = image_points[:, 0] >= xmargin_l
        idx_visible = idx_visible[_valid_]
        image_points = image_points[_valid_]
        if idx_visible.shape[0] == 0: return pd
        # Discard points with image x-coordinate > width-margin
        _valid_ = image_points[:, 0] <= (self.width-xmargin_r)
        idx_visible = idx_visible[_valid_]
        image_points = image_points[_valid_]
        if idx_visible.shape[0] == 0: return pd
        # Discard points with image y-coordinate < margin
        _valid_ = image_points[:, 1] >= ymargin_t
        idx_visible = idx_visible[_valid_]
        image_points = image_points[_valid_]
        if idx_visible.shape[0] == 0: return pd
        # Discard points with image x-coordinate > height-margin
        _valid_ = image_points[:, 1] <= (self.height-ymargin_b)
        idx_visible = idx_visible[_valid_]
        image_points = image_points[_valid_]
        if idx_visible.shape[0] == 0: return pd
        pd[idx_visible] = self.pd
        
#        # Frustum planes
#        # Scale pd according to distance from the plane
#        scale_factor = self.pd*np.ones(idx_visible.shape[0], dtype=np.float)
#        for _normal_ in self._normals_:
#            dot_product = np.dot(rel_points1[idx_visible], _normal_)
#            valid_idx = dot_product < margin
#            idx_visible = idx_visible[valid_idx]
#            scale_factor = scale_factor[valid_idx]
#            dot_product = np.abs(dot_product[valid_idx])
#            scale_idx = dot_product < margin
#            scale_factor[scale_idx] *= dot_product[scale_idx]/margin
#            if idx_visible.shape[0] == 0: return pd
#        pd[idx_visible] = scale_factor
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
        pd = self._pdf_detection_(points1, points2, **kwargs).astype(np.bool)
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points1.shape[0])
        clutter_pdf[pd == False] = 1
        return clutter_pdf
    
    def set_tf_frame(self, frame_id, *args):
        self.tfFrame = frame_id
        self.tf_frame = frame_id
    
    def get_tf_frame(self):
        return (self.tfFrame,)
    

class StereoCameraModel(ros_cameramodels.StereoCameraModel, _FoV_):
    def __init__(self):
        """Idealised stereo camera"""
        self.left = PinholeCameraModel()
        self.right = PinholeCameraModel()
        _FoV_.__init__(self)
        self.tfFrame = None
        self._camera_info_ = None
    
    def copy(self):
        new_camera = StereoCameraModel()
        if not self._camera_info_ is None:
            new_camera.fromCameraInfo(self._camera_info_[0], self._camera_info_[1])
        new_camera.set_tf_frame(self.left.tfFrame, self.right.tfFrame)
        return new_camera
    
    def fromCameraInfo(self, left_msg, right_msg):
        self._camera_info_ = (left_msg, right_msg)
        super(StereoCameraModel, self).fromCameraInfo(left_msg, right_msg)
        self.tfFrame = left_msg.header.frame_id
    
    def camera_matrix(self):
        return self.left.camera_matrix()
    
    def projection_matrix(self):
        return np.vstack((self.left.projection_matrix()[np.newaxis], 
                          self.right.projection_matrix()[np.newaxis]))
    
    def get_width(self):
        return self.left.get_width()
    
    def get_height(self):
        return self.left.get_height()
    
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
        # Convert points to left reference frame
        rel_points = self.left.from_world_coords(points)[0]
        l_pdf = self.left._pdf_detection_(rel_points, **kwargs)
        r_pdf = self.right._pdf_detection_(rel_points, **kwargs)
        l_pdf = np.vstack((l_pdf, r_pdf))
        return np.min(l_pdf, axis=0)
    
    def is_visible(self, points, **kwargs):
        return (self.pdf_detection(points, **kwargs) > 0).astype(np.bool)
    
    def is_visible_relative2sensor(self, rel_points1, rel_points2=None, **kwargs):
        if rel_points1.shape[0] == 0:
            return np.empty(0, dtype=np.bool)
        l_pdf = self.left._pdf_detection_(rel_points1, None, **kwargs)
        if not rel_points2 is None:
            r_pdf = self.right._pdf_detection_(rel_points2, None, **kwargs)
            l_pdf = np.vstack((l_pdf, r_pdf))
            return (np.min(l_pdf, axis=0) > 0).astype(np.bool)
        else:
            return (l_pdf > 0).astype(np.bool)
    
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
    
    def get_tf_frame(self):
        return (self.left.tfFrame, self.right.tfFrame)
    
    def observations(self, states):
        rel_states = self.from_world_coords(states)
        return rel_states

class _CameraFeatureDetector_(object):
    def __init__(self, feature_extractor=image_feature_extractor.Orb, **kwargs):
        self._flann_matcher_ = None
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
    
    def get_features(self, image, mask=None):
        (keypoints, descriptors) = self._featuredetector_.get_features(image, mask)
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
                                       ransacReprojThreshold=3.0,
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
    
    def make_grid_adapted_detector(self):
        self._featuredetector_.make_grid_adapted()
    
    def set_detector_num_features(self, num_features):
        self._featuredetector_.set_num_features(num_features)
    
    def get_detector_num_features(self):
        return self._featuredetector_.get_num_features()
    

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
    
    def points3d_from_img(self, image_left, image_right, ratio_threshold=0.75,
                          image_margins=(0, 0, 0, 0)):
        """get_points(self, image_left, image_right, ratio_threshold=0.6)
            -> points3d, keypoints, descriptors
        """
        images = [STRUCT(), STRUCT()]
        # Get keypoints and descriptors from images
        """
        for (idx, _im_) in zip((0, 1), (image_left, image_right)):
            #self.images[idx].raw = _im_.copy()
            (images[idx].keypoints, images[idx].descriptors) = (
            self.get_features(_im_))
        """
        margin_l, margin_r, margin_t, margin_b = image_margins
        mask_margins = np.any(image_margins)
        (images[0].keypoints, images[0].descriptors) = (
            self.get_features(image_left))
        if mask_margins:
            this_img_keypoints = images[0].keypoints
            this_img_descriptors = images[0].descriptors
            _img_pts_ = np.asarray([_kp_.pt for _kp_ in this_img_keypoints])
            valid_idx = np.ones(_img_pts_.shape[0]).astype(np.bool)
            valid_idx[_img_pts_[:, 0] < margin_l] = False
            valid_idx[_img_pts_[:, 0] > (image_left.shape[1]-margin_r)] = False
            valid_idx[_img_pts_[:, 1] < margin_t] = False
            valid_idx[_img_pts_[:, 1] > (image_left.shape[0]-margin_b)] = False
            valid_idx = np.where(valid_idx)[0]
            _masked_keypoints_ = [this_img_keypoints[_idx_] for _idx_ in valid_idx]
            _masked_descriptors_ = this_img_descriptors[valid_idx]
            images[0].keypoints = _masked_keypoints_
            images[0].descriptors = _masked_descriptors_
        
        try:
            num_features = self.get_detector_num_features()
        except UnboundLocalError:
            pass
        try:
            try:
                self.set_detector_num_features(np.min([num_features*10, 2000]))
            except UnboundLocalError:
                pass
            """
            for (idx, _im_) in zip((0, 1), (image_left, image_right)):
                #self.images[idx].raw = _im_.copy()
                (images[idx].keypoints, images[idx].descriptors) = (
                self.get_features(_im_))
            """
            (images[1].keypoints, images[1].descriptors) = (
                self.get_features(image_right))
            if mask_margins:
                this_img_keypoints = images[1].keypoints
                this_img_descriptors = images[1].descriptors
                _img_pts_ = np.asarray([_kp_.pt for _kp_ in this_img_keypoints])
                valid_idx = np.ones(_img_pts_.shape[0]).astype(np.bool)
                valid_idx[_img_pts_[:, 0] < margin_l] = False
                valid_idx[_img_pts_[:, 0] > (image_left.shape[1]-margin_r)] = False
                valid_idx[_img_pts_[:, 1] < margin_t] = False
                valid_idx[_img_pts_[:, 1] > (image_left.shape[0]-margin_b)] = False
                valid_idx = np.where(valid_idx)[0]
                _masked_keypoints_ = [this_img_keypoints[_idx_] for _idx_ in valid_idx]
                _masked_descriptors_ = this_img_descriptors[valid_idx]
                images[1].keypoints = _masked_keypoints_
                images[1].descriptors = _masked_descriptors_
            # Use the flann matcher to match keypoints
            im_left = images[0]
            im_right = images[1]
            if (len(im_left.keypoints) > 2) and (len(im_right.keypoints) > 2):
                match_result = self._flann_matcher_.detect_and_match(
                    im_left.keypoints, im_left.descriptors,
                    im_right.keypoints, im_right.descriptors, ratio_threshold)
                pts_l, pts_r, idx_l, idx_r, mask_tuple = match_result
            else:
                pts_l = np.empty(0)
                pts_r = np.empty(0)
            
            # Only proceed if there are matches
            if pts_l.shape[0]:
                #print "matches found"
                kp_l = np.asarray(im_left.keypoints)[idx_l]
                desc_l = np.asarray(im_left.descriptors)[idx_l]
                kp_r = np.asarray(im_right.keypoints)[idx_r]
                desc_r = np.asarray(im_right.descriptors)[idx_r]
                # Valid matches are those where y co-ordinate of p1 and p2 are
                # almost equal
                # Subtract offset from inaccurate calibration
                #pts_r[:, 1] += 39.9
                y_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
                #print "Average y_diff = ", np.min(y_diff), np.mean(y_diff), np.max(y_diff)
                #print "y_diff std = ", np.std(y_diff)
                valid_disparity_mask = y_diff < 3.0
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
        except:
            exc_info = sys.exc_info()
            print "STEREOCAMERAFEATUREDETECTOR: POINTS3D_FROM_IMG():"
            print traceback.print_tb(exc_info[2])
            
        finally:
            try:
                self.set_detector_num_features(num_features)
            except UnboundLocalError:
                pass
        #print "Found (%s, %s) keypoints" % (len(images[0].keypoints), len(images[1].keypoints))
        return points3d, (pts_l, pts_r), (kp_l, kp_r), (desc_l, desc_r)
    
