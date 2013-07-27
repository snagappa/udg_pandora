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
# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
#import std_msgs.msg
from sensor_msgs.msg import CameraInfo
import hwu_meta_data.metaclient as metaclient
import numpy as np
from lib.misctools import STRUCT, image_converter, camera_buffer #, Retinex
from lib import image_feature_extractor
from lib.phdfilter import GMPHD
import tf
from tf import TransformBroadcaster, TransformListener
from tf.transformations import quaternion_from_euler, euler_from_quaternion, \
    quaternion_matrix, euler_matrix
from geometry_msgs.msg import PoseWithCovarianceStamped
import itertools
from lib.cameramodels import transform_numpy_array
from collections import deque


class phdmapper:
    def __init__(self, name):
        self.name = name
        
        # Current pose of vehicle
        vehicle_pose = STRUCT()
        vehicle_pose.position = np.zeros(3)
        vehicle_pose.orientation_rpy = np.zeros(3)
        vehicle_pose.covariance = np.zeros((6, 6))
        self.vehicle_pose = vehicle_pose
        
        # Load ROS params for the PHD filter
        
        # Create GM PHD instance
        self.phd_map = phdfilter.GMPHD()
        
        # Create callback to update vehicle position as published by
        # pose EKF slam node
        
        # Create callback to generate map using points published by
        # slam_feature_detector node
        
    
    def update_features(self, pcl_msg):
        init = self.config.init
        if (not init.init) or (not init.dvl) or (not init.map):
            print "Not initialised, not updating features"
            return
        
        self.__LOCK__.acquire()
        try:
            self.config.last_time.map = copy.copy(pcl_msg.header.stamp)
            self.predict(self.config.last_time.map)
            # Convert the pointcloud slam features into normal x,y,z
            # The pointcloud may have uncertainty on the points - this will be
            # the observation noise
            #slam_features = pointclouds.pointcloud2_to_xyz_array(pcl_msg)
            slam_features = self.ros.pcl_helper.from_pcl(pcl_msg)
            # We can now access the points as slam_features[i]
            self.slam_worker.update_features(slam_features)
            eff_nparticles = 1/np.power(self.slam_worker.vehicle.weights, 2).sum()
            if eff_nparticles < 1.5:
                # Shift the states so that the central state is equal to the mean
                state_est = self.slam_worker.estimate()
                state_delta = (self.slam_worker.vehicle.states[0, 0:3] - 
                               state_est.vehicle.ned.state)
                self.slam_worker.vehicle.states[:, 0:3] += state_delta
                max_wt_idx = self.slam_worker.vehicle.weights.argmax()
                self.slam_worker.vehicle.maps = [
                    self.slam_worker.vehicle.maps[max_wt_idx].copy()
                    for count in range(self.slam_particles)]
                print "Moving states..."
                print "State Delta = ", state_delta
                # Copy the particle state to the PHD parent state
                parent_ned = np.array(self.slam_worker.vehicle.states[:, 0:3])
                self.slam_worker._copy_state_to_map_(parent_ned)
                # All particles now have equal weight
                nparticles = self.slam_worker.vars.nparticles
                self.slam_worker.vehicle.weights = (
                    1.0/nparticles*np.ones(nparticles))
            self.ros.last_update_time = pcl_msg.header.stamp
        except:
            print "Error occurred while updating features"
            exc_info = sys.exc_info()
            print "G500_SLAM:UPDATE_FEATURES():\n", traceback.print_tb(exc_info[2])
            traceback.print_tb(exc_info[2])
        else:
            # Republish the observations with slam_sensor0 as the frame_id
            try:
                if slam_features.shape[0]:
                    world_slam_features = self.slam_worker.vehicle.maps[0].sensors.camera.to_world_coords(np.asarray(slam_features[:, :3], order='C'))
                    world_obs_pcl_msg = self.ros.pcl_helper.to_pcl(np.hstack((world_slam_features, slam_features[:, 3:])))
                    world_obs_pcl_msg.header.stamp = pcl_msg.header.stamp
                    world_obs_pcl_msg.header.frame_id = "world"
                    self.ros.map.obs_publisher.publish(world_obs_pcl_msg)
                else:
                    pcl_msg.header.frame_id = "world"
                    self.ros.map.obs_publisher.publish(pcl_msg)
            except:
                print "Error republishing features..."
        finally:
            self.__LOCK__.release()
    
    def publish_data(self, *args, **kwargs):
        if not self.config.init.init:
            print "Not initialised yet..."
            # self.ros.last_update_time = rospy.Time.now()
            # self.config.init = True
            return
        nav_msg = self.ros.nav.msg
        slam_estimate = self.slam_worker.estimate()
        est_state = slam_estimate.vehicle.ned.state
        est_cov = slam_estimate.vehicle.ned.covariance
        est_state_vel = slam_estimate.vehicle.vel_xyz.state
        angle = tf.transformations.euler_from_quaternion(
            self.vehicle.pose_orientation)
        
        # Create header
        timestamp = self.ros.last_update_time
        nav_msg.header.stamp = timestamp
        nav_msg.header.frame_id = self.ros.name
        world_frame_id = "world"
                   
        #Fill Nav status topic
        nav_msg.position.north = est_state[0]
        nav_msg.position.east = est_state[1]
        nav_msg.position.depth = est_state[2]
        nav_msg.body_velocity.x = est_state_vel[0]
        nav_msg.body_velocity.y = est_state_vel[1]
        nav_msg.body_velocity.z = est_state_vel[2]
        nav_msg.orientation.roll = angle[0]
        nav_msg.orientation.pitch = angle[1]
        nav_msg.orientation.yaw = angle[2]
        nav_msg.orientation_rate.roll = self.vehicle.twist_angular[0]
        nav_msg.orientation_rate.pitch = self.vehicle.twist_angular[1]
        nav_msg.orientation_rate.yaw = self.vehicle.twist_angular[2]
        nav_msg.altitude = self.vehicle.altitude
        
        # Variance
        nav_msg.position_variance.north = est_cov[0, 0]
        nav_msg.position_variance.east = est_cov[1, 1]
        nav_msg.position_variance.depth = est_cov[2, 2]
        
        #nav_msg.status = np.uint8(np.log10(self.ros.NO_LOCK_ACQUIRE+1))
        self.ros.err_pub.publish(std_msgs.msg.Int32(self.ros.NO_LOCK_ACQUIRE))
        
        #Publish topics
        self.ros.nav.publisher.publish(nav_msg)
        
        #Publish TF for NavSts
        platform_orientation = self.vehicle.pose_orientation
        br = tf.TransformBroadcaster()
        br.sendTransform((nav_msg.position.north, 
                          nav_msg.position.east, nav_msg.position.depth),
                         platform_orientation, timestamp, 
                         nav_msg.header.frame_id, world_frame_id)
        
        # Publish landmarks now
        map_estimate = slam_estimate.map
        map_states = map_estimate.state
        map_covs = map_estimate.covariance
        if map_states.shape[0]:
            diag_cov = np.array([np.diag(map_covs[i]) 
                for i in range(map_covs.shape[0])])
            pcl_msg = self.ros.map.helper.to_pcl(
                np.hstack((map_states, diag_cov)))
        else:
            pcl_msg = self.ros.map.helper.to_pcl(np.zeros(0))
        pcl_msg.header.stamp = timestamp
        pcl_msg.header.frame_id = world_frame_id
        self.ros.map.publisher.publish(pcl_msg)
        
        
        #print "Landmarks at: "
        #print map_states
        
        #print "Tracking ", map_estimate.weight.shape[0], \
        #    " (", map_estimate.weight.sum(), ") targets."
        #print "Intensity = ", map_estimate.weight.sum()
        #dropped_msg_time = \
        #    (rospy.Time.now()-self.config.last_time.init).to_sec()
        #print "Dropped ", self.ros.NO_LOCK_ACQUIRE, " messages in ", \
        #    int(dropped_msg_time), " seconds."