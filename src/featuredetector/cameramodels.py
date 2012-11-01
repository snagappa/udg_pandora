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
from lib.common.misctools import STRUCT

default_float = "float64"

def _raise_(ex):
    raise ex


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
        try:
            self.tflistener = tf.TransformListener()
        except (rospy.ROSInitException, rospy.ROSException, 
                rospy.ServiceException) as tferror:
            self.tferror = tferror
            self.tflistener = STRUCT()
            self.tflistener.transformPointCloud = (
                lambda *args, **kwargs: _raise_(self.tferror))
            print ("Error initialising tflistener, transforms are unavailable")
            print tferror
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=5.0):
        """set_near_far_fov(self, fov_near=0.3, fov_far=5.0) -> None
        Set the near and far distances visible by the camera
        """
        self.fov_near = fov_near
        self.fov_far = fov_far
        self.observation_volume = self._observation_volume_()
    
    def _observation_volume_(self):
        return 4/3*np.pi*(self.fov_far**3 - self.fov_near**3)
    
    def set_const_pd(self, pd=0.9):
        """set_const_pd(self, pd=0.9) -> None
        Set the probability of target detection for the model
        """
        self.pd = pd
    
    def pdf_detection(self, points):
        """pdf_detection(self, points) -> pd
        Returns the probability of detection for points between the spheres of
        radius fov_near and fov_far.
        """
        euclid_distance = np.power(np.power(points, 2).sum(axis=1), 0.5)
        pd = np.zeros(points.shape[0])
        pd[np.logical_and(self.fov_near < euclid_distance, 
                          euclid_distance < self.fov_far)] = self.pd
        return pd
    
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
        pcl_tf_points = self.tflistener.transformPointCloud(target_frame, 
                                                            pcl_msg)
        if RETURN_NP_ARRAY:
            return self.pcl_helper.from_pcl(pcl_tf_points)
        else:
            return pcl_tf_points
    
    def _nparray_to_pcl_(self, numpy_points, frame_id):
        assert (((numpy_points.ndim == 2) and (numpy_points.shape[1] == 3)) or
            (np.prod(numpy_points.shape == 0))), (
            "Points must be Nx3 numpy array")
        pcl_points = self.pcl_helper.to_pcl(numpy_points)
        pcl_points.header.stamp = rospy.Time.now()
        pcl_points.header.frame_id = frame_id
        return pcl_points
        
    def to_world_coords(self, numpy_points):
        """to_world_coords(self, numpy_points)->world_points
        Convert Nx3 numpy array of points from camera coordinate system to
        world coordinates"""
        pcl_points = self._nparray_to_pcl_(numpy_points, self.tfFrame)
        return self.to_world_coords_pcl(pcl_points, True)
    
    def to_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False):
        """to_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False)
        -> world_points OR world_points_pcl
        Convert pointcloud from camera coordinate system to world
        coordinates.
        Set RETURN_NP_ARRAY to True to return a numpy array instead of the
        pointcloud"""
        target_frame = "world"
        return self._perform_tf_(target_frame, pcl_msg, RETURN_NP_ARRAY)
    
    def from_world_coords(self, numpy_points):
        """from_world_coords(self, points)->camera_points
        Convert Nx3 numpy array of points from world coordinate system to
        camera coordinates"""
        pcl_points = self._nparray_to_pcl_(numpy_points, "world")
        return self.from_world_coords_pcl(pcl_points, True)
    
    def from_world_coords_pcl(self, pcl_msg, RETURN_NP_ARRAY=False):
        """from_world_coords_pcl(self, pcl, RETURN_NP_ARRAY=False)
        ->camera_points OR camera_pcl
        Convert points from world coordinate system to camera coordinates.
        Set RETURN_NP_ARRAY to True to return a numpy array instead of the
        pointcloud"""
        target_frame = self.tfFrame
        return self._perform_tf_(target_frame, pcl_msg, RETURN_NP_ARRAY)
    

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
        dst = blas.dgemv(P, src)
        
        pixels = np.asarray(dst[:, :2], order='C')
        w = dst[:, 2]
        w[w == 0] = np.inf
        pixels /= w[:, np.newaxis]
        return pixels
    
    def _observation_volume_(self):
        """_observation_volume_(self) -> obs_volume
        Calculate the volume of the observation space
        """
        corners_2d = [(0, 0), (self.width, self.height)]
        unit_corners_3d = [np.asarray(self.projectPixelTo3dRay(_corner_))
                           for _corner_ in corners_2d]
        
        scalefactor = np.abs(self.fov_far/unit_corners_3d[0, 2])
        corners_3d = unit_corners_3d*scalefactor
        base_width, base_height = corners_3d[1] - corners_3d[0]
        obs_volume = 0.33*base_width*base_height*self.fov_far
        return obs_volume
    
    def pdf_detection(self, points):
        pd = self.pd*np.ones(points.shape[0], dtype=np.float)
        pd[points[:, 2] < self.fov_near] = 0
        pd[points[:, 2] > self.fov_far] = 0
        pixels = self.project3dToPixel(points)
        px_margin = 1e-2
        pd[pixels[:, 0] < px_margin] = 0
        pd[pixels[:, 0] > (self.width+px_margin)] = 0
        pd[pixels[:, 1] < px_margin] = 0
        pd[pixels[:, 1] > (self.height+px_margin)] = 0
        return pd
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = bool(self.pdf_detection(points))
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points.shape[0])
        clutter_pdf[pd == False] = 1
        return 
    

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
    
    def pdf_detection(self, points):
        """pdf_detection(self, points) -> pd
        Calculate the probability of detection for the points
        """
        l_pdf = self.left.pdf_detection(points)
        r_pdf = self.left.pdf_detection(points)
        l_pdf[r_pdf == 0] = 0
        return l_pdf
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the pdf of the clutter at the given points
        """
        return self.left.pdf_clutter(points)
    
