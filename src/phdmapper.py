#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       phdmapper.py
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

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import PointCloud2
# Normal python imports
import numpy as np
import copy
import sys, traceback
# Custom imports
from lib.misctools import STRUCT, message_buffer, camera_buffer, pcl_xyz_cov
from lib import phdfilter
from tf.transformations import quaternion_from_euler
from tf import TransformBroadcaster

PHDMAPPER_ROOT = "phdmapper/"
PHDFILTER_ROOT = PHDMAPPER_ROOT + "filter/"


def get_default_parameters():
    """get_default_parameters() --> parameters
    Returns the default parameters for the PHD mapper.
    """
    parameters = {}
    phdmapper_root = PHDMAPPER_ROOT
    phdfilter_root = PHDFILTER_ROOT
    # Parameters for the PHD filter
    parameters[phdfilter_root+"prune_threshold"] = 0.0025
    parameters[phdfilter_root+"merge_threshold"] = 0.75
    parameters[phdfilter_root+"max_num_components"] = 8000
    parameters[phdfilter_root+"birth_intensity"] = 0.05
    parameters[phdfilter_root+"clutter_intensity"] = 10.
    parameters[phdfilter_root+"detection_probability"] = 0.9
    # Parameters for the mapping mode
    parameters[phdmapper_root+"use_3d"] = False
    parameters[phdmapper_root+"use_ukf"] = True
    return parameters

def set_default_parameters():
    """set_default_parameters()
    Sets the default parameters for the PHD mapper.
    """
    parameters = get_default_parameters()
    for _param_name_ in parameters:
        rospy.set_param(_param_name_, parameters[_param_name_])


class PHDMAPPER(object):
    """PHDMAPPER(object)
    Wrapper class for performing mapping using the PHD filter with pointcloud
    observations.
    """
    def __init__(self, name):
        """PHDMAPPER(ROS_NODE_NAME) --> phdmapper
        Create a PHDMAPPER object to perform mapping using the PHD filter.
        """
        self.name = name
        
        # Load ROS params for the PHD filter
        # Filter parameters
        self.phdfilter_config = None
        # Options for the mapping
        self.mapper_config = None
        self.read_config()
        
        # Create GM PHD instance
        self.phd_map = phdfilter.GMPHD(self.phdfilter_config)
        # Initialise the estimate to the null set
        self.map_estimate = self.phd_map.estimate()
        self.last_update_time = rospy.Time(0)
        # Convert Pointcloud messages to numpy arrays
        self.pcl_convertor = pcl_xyz_cov()
        # Publisher for the pointcloud message
        self.publisher = rospy.Publisher("/visual_detector/phdmap", PointCloud2)
        
        # Get {left,right}/camera_info
        camera_root = rospy.get_param("slam_feature_detector/camera_root")
        image_topic = rospy.get_param("slam_feature_detector/image_topic")
        
        camera = camera_buffer(camera_root, image_topic, IS_STEREO=True)
        camera_info_l, camera_info_r = camera.get_camera_info()
        # Initialise the camera in the map - this sets up:
        # 1. the frame id for the tf to be used
        # 2. the transformations between 3D state space and observation space
        self.phd_map.fromCameraInfo(camera_info_l, camera_info_r)
        
        if self.mapper_config.use_3d:
            msg_topic = "features"
        else:
            msg_topic = "disparity"
        self.features_buffer = message_buffer(
            ["/slam_feature_detector/"+msg_topic], [PointCloud2])
        # Create callback to generate map using points published by
        # slam_feature_detector node
        self.callback_id = self.features_buffer.register_callback(
            self.update_features)
        
        # Get camera details
        self.tf_info = STRUCT()
        cam_info_msg = camera.get_camera_info(0)[0]
        self.tf_info.frame_id = cam_info_msg.header.frame_id
        self.tf_info.position = tuple(rospy.get_param(
            "phdmapper/camera_position"))
        self.tf_info.orientation = quaternion_from_euler(
            *rospy.get_param("phdmapper/camera_orientation"))
        self.tf_info.baseline = tuple(rospy.get_param(
            "phdmapper/camera_baseline"))
        self.tf_info.baseline_orientation = quaternion_from_euler(
            *rospy.get_param("phdmapper/camera_baseline_orientation"))
        self.tf_broadcaster = TransformBroadcaster()
        rospy.timer.Timer(rospy.Duration(0.1), self.publish_transforms)
        print "Completed initialisation."
    
    def publish_transforms(self, *args, **kwargs):
        timestamp = rospy.Time.now()
        tf_br = self.tf_broadcaster
        tf_info = self.tf_info
        frame_id = tf_info.frame_id
        
        # Don't publish the left frame if already being published
#        tf_br.sendTransform(tf_info.position, tf_info.orientation,
#                            timestamp, frame_id, 'girona500')
        tf_br.sendTransform(tf_info.baseline, tf_info.baseline_orientation,
                            timestamp, frame_id+'_right', frame_id)
    
    def read_config(self):
        """PHDMAPPER.read_config()
        Reads the configuration from the ROS parameter server.
        """
        phdmapper_root = PHDMAPPER_ROOT
        phdfilter_root = PHDFILTER_ROOT
        phdfilter_config = STRUCT()
        # Prune components less than this weight
        phdfilter_config.prune_threshold = rospy.get_param(
            phdfilter_root+"prune_threshold")
        # Merge components  closer than this threshold
        phdfilter_config.merge_threshold = rospy.get_param(
            phdfilter_root+"merge_threshold")
        # Maximum number of components to track
        phdfilter_config.max_num_components = rospy.get_param(
            phdfilter_root+"max_num_components")
        # Intensity of new targets
        phdfilter_config.birth_intensity = rospy.get_param(
            phdfilter_root+"birth_intensity")
        # Intensity of clutter in the scene
        phdfilter_config.clutter_intensity = rospy.get_param(
            phdfilter_root+"clutter_intensity")
        # Probability of detection of targets in the FoV
        phdfilter_config.pd = rospy.get_param(
            phdfilter_root+"detection_probability")
        self.phdfilter_config = phdfilter_config
        
        mapper_config = STRUCT()
        # Whether to use 3D features or disparity features
        mapper_config.use_3d = rospy.get_param(phdmapper_root+"use_3d")
        # Whether to use the UKF prediction/update whenever possible
        mapper_config.use_ukf = rospy.get_param(phdmapper_root+"use_ukf")
        self.mapper_config = mapper_config
    
    def update_features(self, pcl_msg):
        """PHDMAPPER.update_features(pointcloud2_message)
        Updates the map using the observations from the pointcloud message.
        Each entry in the pointcloud should be of the form:
            [x, y, z, cov_x, cov_y, cov_z]
        If use_3d is set to False in the configuration, z corresponds to the
        disparity between the left and right images.
        """
        try:
            self.last_update_time = copy.copy(pcl_msg.header.stamp)
            # Convert the pointcloud slam features into normal x,y,z
            # The pointcloud may have uncertainty on the points - this will be
            # the observation noise
            #slam_features = pointclouds.pointcloud2_to_xyz_array(pcl_msg)
            slam_features = self.pcl_convertor.from_pcl(pcl_msg)
            # We can now access the points as slam_features[i]
            self.phd_map.update_features(slam_features, self.last_update_time,
                self.mapper_config.use_3d, self.mapper_config.use_ukf,
                DISPLAY=False)
            self.map_estimate = self.phd_map.estimate()
            rospy.loginfo("Updated map with %d features" % slam_features.shape[0])
            rospy.loginfo("Map size: %d points" % self.map_estimate.state.shape[0])
        except:
            print "Error occurred in"
            exc_info = sys.exc_info()
            print "phdmapper:UPDATE_FEATURES\n", traceback.print_tb(exc_info[2])
            traceback.print_tb(exc_info[2])
        else:
            self.publish_map()
    
    def publish_map(self, *args, **kwargs):
        """PHDMAPPER.publish_map()
        Publish the map as a pointcloud message to "/visual_detector/phdmap".
        """
        # Publish landmarks now
        map_estimate = self.map_estimate
        map_states = map_estimate.state
        map_covs = map_estimate.covariance
        if map_states.shape[0]:
            diag_cov = np.array([np.diag(map_covs[i]) 
                for i in range(map_covs.shape[0])])
            pcl_map_points = np.hstack((map_states, diag_cov))
        else:
            pcl_map_points = np.zeros(0)
        pcl_msg = self.pcl_convertor.to_pcl(pcl_map_points)
        pcl_msg.header.stamp = self.last_update_time
        pcl_msg.header.frame_id = "world"
        self.publisher.publish(pcl_msg)
    


if __name__ == '__main__':
    try:
        import subprocess
        # Load ROS parameters
        config_file_list = roslib.packages.find_resource("udg_pandora",
            "phdmapper.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            print "Could not locate visual_detector.yaml, using defaults"
            set_default_parameters()
        
        rospy.init_node('visual_detector')
        pdhmapper_node = PHDMAPPER(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
