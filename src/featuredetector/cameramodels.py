# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:20:27 2012

@author: snagappa
"""

from image_geometry import cameramodels as ros_cameramodels
import numpy as np
from lib.common import blas
import tf

class PinholeCameraModel(ros_cameramodels.PinholeCameraModel):
    def __init__(self):
        """
        Pinhole camera model derived from image_geometry.PinholeCameraModel
        Additional methods to estimate detection probability and clutter pdf.
        """
        super(PinholeCameraModel, self).__init__()
        self.pd = 0.9
        self.fov_near = 0.3
        self.fov_far = 5.0
        self.tfFrame = None
        self.observation_volume = 0.0
    
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
        w[w==0] = np.inf
        pixels /= w[:, np.newaxis]
        return pixels
    
    def set_near_far_fov(self, fov_near=0.3, fov_far=5.0):
        """set_near_far_fov(self, fov_near=0.3, fov_far=5.0) -> None
        Set the near and far distances visible by the camera
        """
        self.fov_near = fov_near
        self.fov_far = fov_far
        self.observation_volume = self._observation_volume_()
    
    def set_const_pd(self, pd=0.9):
        """set_const_pd(self, pd=0.9) -> None
        Set the probability of target detection for the model
        """
        self.pd = pd
    
    def pdf_detection(self, points):
        """pdf_detection(self, points) -> pd
        Returns the probability of detection for points in the camera
        coordinate system.
        """
        pd = self.pd*np.ones(points.shape[0], dtype=np.float)
        pd[points[:, 2]<self.fov_near] = 0
        pd[points[:, 2]>self.fov_far] = 0
        pixels = self.project3dToPixel(points)
        px_margin = 1e-2
        pd[pixels[:, 0]<px_margin] = 0
        pd[pixels[:, 0]>(self.width+px_margin)] = 0
        pd[pixels[:, 1]<px_margin] = 0
        pd[pixels[:, 1]>(self.height+px_margin)] = 0
        return pd
    
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
        
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the probability distribution for the clutter at the given
        points
        """
        pd = bool(self.pdf_detection(points))
        clutter_pdf = (1.0/self.observation_volume)*np.ones(points.shape[0])
        clutter_pdf[pd==False] = 1
        return 
    
    def to_world_coords(self, numpy_points, frame_id="world"):
        """Convert points from camera coordinate system to world"""
        pass
    
    def to_world_coords_pcl(self, pcl, frame_id="world"):
        """Convert pointcloud from camera coordinate system to world"""
        pass
    
    def from_world_coords(self, points):
        """Convert points from world coordinate system to camera"""
        pass
    
    def from_world_coords_pcl(self, pcl):
        """Convert points from world coordinate system to camera"""
        pass
    

class StereoCameraModel(ros_cameramodels.StereoCameraModel):
    def __init__(self):
        """Idealised stereo camera"""
        self.left = PinholeCameraModel()
        self.right = PinholeCameraModel()
    
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
        l_pdf[r_pdf==0] = 0
        return l_pdf
    
    def pdf_clutter(self, points):
        """pdf_clutter(self, points) -> clutter_pdf
        Calculate the pdf of the clutter at the given points
        """
        return self.left.pdf_clutter(points)
    
