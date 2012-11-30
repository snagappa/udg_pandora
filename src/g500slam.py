#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       ros_slam.py
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
try:
    roslib.load_manifest("navigation_g500")
except:
    pass
import rospy
import tf
import PyKDL
import math
#import code
import copy
import threading
import sys
import pickle
import cola2_navigation.msg
try:
    import navigation_g500.msg
except:
    pass
import visual_detector
import traceback

USE_COLA_TOPICS =  True
USE_GPS = False
SIMULATOR = True

CAMERA_ROOT = "/stereo_down"
CAMERA_ORIENTATION = "down"

nav_sensors_msg = cola2_navigation.msg
SENSOR_TOPIC_ROOT = "/cola2_navigation/"
if not USE_COLA_TOPICS:
    try:
        nav_sensors_msg = navigation_g500.msg
    except:
        print "navigation_g500.msg is unavailable, using cola2"
    else:
        SENSOR_TOPIC_ROOT = "/navigation_g500/"

TeledyneExplorerDvl = nav_sensors_msg.TeledyneExplorerDvl
ValeportSoundVelocity = nav_sensors_msg.ValeportSoundVelocity
FastraxIt500Gps = nav_sensors_msg.FastraxIt500Gps
    
from sensor_msgs.msg import Imu, PointCloud2, CameraInfo, std_msgs
from auv_msgs.msg import NavSts
from std_srvs.srv import Empty, EmptyResponse
from cola2_navigation.srv import SetNE, SetNEResponse #, SetNERequest

# import pyximport; pyximport.install()
import lib.slam_worker
#from lib.slam_worker import PHDSLAM
PHDSLAM = lib.slam_worker.PHDSLAM
import numpy as np
from lib.common.ros_helper import get_g500_config
from lib.common.kalmanfilter import sigma_pts
from lib.common import misctools
from lib.common.misctools import STRUCT
from featuredetector import cameramodels

INVALID_ALTITUDE = -32665
SAVITZKY_GOLAY_COEFFS = [0.2,  0.1,  0. , -0.1, -0.2]

UKF_ALPHA = 0.2
UKF_BETA = 1
UKF_KAPPA = 0

# Profile options
__PROFILE__ = False
__PROFILE_NUM_LOOPS__ = 100
#code.interact(local=locals())

def normalize_angle(np_array):
    return (np_array + (2.0*np.pi*np.floor((np.pi - np_array)/(2.0*np.pi))))

    
class G500_SLAM():
    def __init__(self, name):
        # Get config
        self.config = get_g500_config()
        
        # Main SLAM worker
        self.slam_particles = 5
        self.slam_worker = self.init_slam()
        try:
            self.create_sigma_states(np.zeros(2))
        except AssertionError as assert_err:
            print assert_err
            rospy.loginfo("Failed to set sigma states")
            self.slam_worker.vehicle.states[:, :] = 0
        
        # Structure to store vehicle pose from sensors
        self.vehicle = STRUCT()
        # roll, pitch and yaw rates - equivalent to odom.twist.twist.angular
        self.vehicle.twist_angular = np.zeros(3, dtype=float)
        self.vehicle.twist_linear = np.zeros(3, dtype=float)
        # position x, y, z - the state space estimated by the filter.
        self.vehicle.pose_position = np.zeros(3, dtype=float)
        # orientation quaternion x,y,z,w - same as odom.pose.pose.orientation
        self.vehicle.pose_orientation = np.zeros(4, dtype=float)
        # Altitude from dvl
        self.vehicle.altitude = 0.0
        
        self.__LOCK__ = threading.Lock()
        # Initialise ROS stuff
        self.init_config()
        self.ros = STRUCT()
        self.init_ros(name)
        
        # Set as true to run without gps/imu initialisation
        self.config.init.init = False
        
        
    def init_slam(self):
        slam_worker = PHDSLAM(nparticles=self.slam_particles)
        slam_worker.set_parameters(
            Q=np.eye(3)*self.config.model_covariance, 
            gpsH=np.hstack(( np.eye(2), np.zeros((2,4)) )), 
            gpsR=np.eye(2)*self.config.gps_covariance, 
            dvlH=np.hstack(( np.zeros((3,3)), np.eye(3) )), 
            dvl_b_R=np.eye(3)*self.config.dvl_bottom_covariance, 
            dvl_w_R=np.eye(3)*self.config.dvl_water_covariance)
        return slam_worker
        
    def create_sigma_states(self, mean_state):
        assert mean_state.ndim == 1, "mean_state must have shape (N)"
        assert mean_state.shape[0] <= self.slam_worker.vars.ndims, (
            "mean_state must have dimension <= %s" % mean_state.shape[0])
        num_sigma_states = 2*mean_state.shape[0] + 1
        assert num_sigma_states == self.slam_particles, (
            "Number of sigma states and particles is incompatible")
        sigma_states = self._make_sigma_states_(self.slam_worker, 
                mean_state[np.newaxis])
        self.slam_worker.set_states(states=sigma_states)
        print "Set new sigma states:\n", self.slam_worker.vehicle.states
    
    def _make_sigma_states_(self, slam_worker, mean_state):
        # Generate covariance
        sc_process_noise = \
            slam_worker.trans_matrices(np.zeros(3), 1.0)[1] + \
            slam_worker.trans_matrices(np.zeros(3), 0.01)[1]
        state_dim = mean_state.shape[1]
        num_sigma_states = 2*state_dim + 1
        # Create sigma states over dimensions specified by mean_state
        sigma_states = (
            sigma_pts(mean_state, 
                      sc_process_noise[0:state_dim, 0:state_dim].copy(), 
                      _alpha=UKF_ALPHA, _beta=UKF_BETA, _kappa=UKF_KAPPA)[0])
        # Pad with zeros to achieve state dimension 6
        sigma_states = np.array(np.hstack((sigma_states, 
            np.zeros((num_sigma_states, 
            self.slam_worker.vars.ndims-state_dim)))), order='C')
        return sigma_states
    
    def init_config(self):
        config = self.config
        
        #Create static transformations
        config.dvl_tf = self.compute_tf(config.dvl_tf_data)
        config.imu_tf = self.compute_tf(config.imu_tf_data)        
        config.svs_tf = self.compute_tf(config.svs_tf_data)
        
        #Initialize flags
        config.init = STRUCT()
        config.init.init = False
        config.init.north = 0.0
        config.init.east = 0.0
        config.init.dvl = False
        config.init.imu = False
        config.init.svs = False
        config.init.map = False
        
        #init last sensor update
        time_now = rospy.Time.now()
        config.last_time = STRUCT()
        config.last_time.init = copy.copy(time_now)
        config.last_time.predict = copy.copy(time_now)
        config.last_time.gps = copy.copy(time_now)
        config.last_time.dvl = copy.copy(time_now)
        config.last_time.imu = copy.copy(time_now)
        config.last_time.svs = copy.copy(time_now)
        config.last_time.map = copy.copy(time_now)
        
        config.imu_data = False
        if USE_GPS:
            config.gps_data = not config.gps_update
        else:
            config.gps_data = True
        config.altitude = INVALID_ALTITUDE
        config.bottom_status = 0
        config.gps_init_samples_list = []
        # Buffer for smoothing the yaw rate
        config.heading_buffer = []
        config.savitzky_golay_coeffs = SAVITZKY_GOLAY_COEFFS
        
    def init_ros(self, name):
        ros = self.ros
        ros.name = name
        self.publish_transforms()
        ros.last_update_time = rospy.Time.now()
        ros.NO_LOCK_ACQUIRE = 0
        
        ros.pcl_helper = misctools.pcl_xyz_cov()
        
        if not __PROFILE__:
            self.ros.subs = STRUCT()
            print "Creating ROS subscriptions..."
            # Create Subscriber
            self.ros.subs.dvl = rospy.Subscriber(
                SENSOR_TOPIC_ROOT+"teledyne_explorer_dvl", 
                TeledyneExplorerDvl, self.update_dvl)
            self.ros.subs.svs = rospy.Subscriber(
                SENSOR_TOPIC_ROOT+"valeport_sound_velocity", 
                ValeportSoundVelocity, self.update_svs)
            self.ros.subs.imu = rospy.Subscriber(SENSOR_TOPIC_ROOT+"imu", 
                                                 Imu, self.update_imu)
            if self.config.gps_update :
                self.ros.subs.gps = rospy.Subscriber(
                    SENSOR_TOPIC_ROOT+"fastrax_it_500_gps", 
                    FastraxIt500Gps, self.update_gps)
            ## Subscribe to visiona slam-features node
            if SIMULATOR:
                print "simulator"
                rospy.Subscriber("/slamsim/features", PointCloud2, 
                                 self.update_features)
            else:
                rospy.Subscriber("/visual_detector2/features", PointCloud2, 
                                 self.update_features)
            # Subscribe to sonar slam features node for
            #rospy.Subscriber("/slam_features/fls_pcl", PointCloud2, 
            #                 self.update_features)
            #Create services
            ros.reset_navigation = \
                rospy.Service('/slam_g500/reset_navigation', 
                              Empty, self.reset_navigation)
            ros.reset_navigation = rospy.Service('/slam_g500/set_navigation', 
                                                 SetNE, self.set_navigation)
            
            # Create publishers
            ros.nav = STRUCT()
            ros.nav.msg = NavSts()
            ros.nav.publisher = rospy.Publisher("/phdslam/nav_sts", NavSts)
            # Publish landmarks
            ros.map = STRUCT()
            ros.map.msg = PointCloud2()
            ros.map.msg.header.frame_id = "world"
            ros.map.publisher = \
                rospy.Publisher("/phdslam/features", PointCloud2)
            ros.map.helper = misctools.pcl_xyz_cov()
            # Republish the observations with slam_base0 as reference
            ros.map.obs_publisher = rospy.Publisher("/phdslam/observations", 
                                                    PointCloud2)
            # Publish count of NO_LOCK_ACQUIRE
            ros.err_pub = rospy.Publisher("/phdslam/NLA", std_msgs.msg.Int32)
            # Publish data every 500 ms
            rospy.timer.Timer(rospy.Duration(0.1), self.publish_data)
            rospy.timer.Timer(rospy.Duration(0.1), self.publish_transforms)
            #rospy.timer.Timer(rospy.Duration(8), self.slam_housekeeping)
            # Callback to print vehicle state and weight
            #rospy.timer.Timer(rospy.Duration(10), self.debug_print)
        else:
            print "** RUNNING IN PROFILER MODE **"
        
        # Initialise the camera field of view
        left_tf_frame = "slam_sensor"
        right_tf_frame = "slam_sensor_right"
        try:
            camera_info_left = rospy.wait_for_message(
                CAMERA_ROOT+"/left/camera_info", CameraInfo, 5)
            camera_info_right = rospy.wait_for_message(
                CAMERA_ROOT+"/right/camera_info", CameraInfo, 5)
        except:
            print "Error occurred initialising camera from camera_info"
            print "Loading camera_info from disk"
            exc_info = sys.exc_info()
            print "G500_SLAM:INIT_ROS():\n", traceback.print_tb(exc_info[2])
            if SIMULATOR:
                camera_info_pickle = (
                    roslib.packages.find_resource("udg_pandora", "camera_info.p"))
            else:
                camera_info_pickle = (
                    roslib.packages.find_resource("udg_pandora", "bumblebee.p"))
            if len(camera_info_pickle):
                camera_info_pickle = camera_info_pickle[0]
                camera_info_left, camera_info_right = (
                    pickle.load(open(camera_info_pickle, "rb")))
            else:
                camera_info_left, camera_info_right = None, None
        #try:
        for _map_, idx in zip(self.slam_worker.vehicle.maps, 
                              range(len(self.slam_worker.vehicle.maps))):
            _map_.sensors.camera.fromCameraInfo(camera_info_left,
                                                camera_info_right)
            str_idx = str(idx)
            _map_.sensors.camera.set_tf_frame(left_tf_frame+str_idx, 
                                              right_tf_frame+str_idx)
            # Set the far field of view
            _map_.sensors.camera.set_near_far_fov(fov_far=visual_detector.FOV_FAR)
        self.config.init.map = True
        """
        except:
            print "Error occurred initialising stereo camera, using dummycamera"
            exc_info = sys.exc_info()
            print "G500_SLAM:INIT_ROS():\n", traceback.print_tb(exc_info[2])
            raise rospy.ROSException("Unable to initialise camera")
        """
        return ros
        
    def reset_navigation(self, req):
        print "Resetting navigation..."
        rospy.loginfo("%s: Reset Navigation", self.ros.name)
        num_sigma_dims = (self.slam_particles-1)/2
        try:
            self.create_sigma_states(np.zeros(num_sigma_dims))
        except AssertionError as assert_err:
            print assert_err
            rospy.loginfo("Failed to set sigma states")
            self.slam_worker.vehicle.states[:, :] = 0
        return EmptyResponse()
        
    def set_navigation(self, req):
        print "Setting new navigation..."
        try:
            self.create_sigma_states(np.array([req.north, req.east]))
        except AssertionError as assert_err:
            print assert_err
            rospy.loginfo("Failed to set sigma states")
            self.slam_worker.vehicle.states[:, :] = 0
        self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
        ret = SetNEResponse()
        ret.success = True
        return ret
    
    def compute_tf(self, transform):
        r = PyKDL.Rotation.RPY(math.radians(transform[3]), 
                               math.radians(transform[4]), 
                               math.radians(transform[5]))
        #rospy.loginfo("Rotation: %s", str(r))
        v = PyKDL.Vector(transform[0], transform[1], transform[2])
        #rospy.loginfo("Vector: %s", str(v))
        frame = PyKDL.Frame(r, v)
        #rospy.loginfo("Frame: %s", str(frame))
        return frame
    
    
    def update_gps(self, gps):
        if USE_GPS and (gps.data_quality >= 1) and (gps.latitude_hemisphere >= 0) and \
        (gps.longitude_hemisphere >= 0):
            print "USE_GPS=%s\ngps.data_quality=%s", (USE_GPS, gps.data_quality)
            config = self.config
            config.last_time.gps = copy.copy(gps.header.stamp)
            if not config.gps_data :
                print "gps not set: initialising"
                config.gps_init_samples_list.append([gps.north, gps.east])
                if len(config.gps_init_samples_list) >= config.gps_init_samples:
                    print "GPS initialised"
                    config.gps_data = True
                    [config.init.north, config.init.east] = \
                            np.median(np.array(config.gps_init_samples_list), 
                                      axis=0)
                    print "Unregistering GPS subscription"
                    self.ros.subs.gps.unregister()
            else:
                config.gps_init_samples_list.pop(0)
                config.gps_init_samples_list.append([gps.north, gps.east])
                [gps_north, gps_east] = np.mean(
                    np.array(config.gps_init_samples_list), axis=0)
                slam_estimate = self.slam_worker.estimate()
                est_state = slam_estimate.vehicle.ned.state[0:2]
                distance = np.sqrt((est_state[0] - gps_north)**2 + 
                                (est_state[1] - gps_east)**2)
                #rospy.loginfo("%s, Distance: %s", self.name, distance)
                
                # Right now the GPS is only used to initialize the navigation 
                # not for updating it!!!
                if distance < 0.5:
                    z = np.array([[gps_north, gps_east]])
                    z_array = self._make_sigma_states_(self.slam_worker, z)[:, :2]
                    self.__LOCK__.acquire()
                    try:
                        if self.predict(config.last_time.gps):
                            self.slam_worker.update_gps(z_array)
                            self.ros.last_update_time = config.last_time.gps
                            self.slam_worker.resample()
                            print "Updated with GPS"
                    except:
                        exc_info = sys.exc_info()
                        print "GPS UPDATE ERROR:\n", traceback.print_tb(exc_info[2])
                    finally:
                        self.__LOCK__.release()
                    #self.publish_data()
                #print "Unregistering GPS subscription"
                #self.ros.subs.gps.unregister()
        
    def update_dvl(self, dvl):
        #print os.getpid()
        config = self.config
        config.last_time.dvl = copy.copy(dvl.header.stamp)
        config.init.dvl = True
        
        # If dvl_update == 0 --> No update
        # If dvl_update == 1 --> Update wrt bottom
        # If dvl_update == 2 --> Update wrt water
        dvl_update = 0
        
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            if (abs(dvl.bi_x_axis) < config.dvl_max_v and 
                abs(dvl.bi_y_axis) < config.dvl_max_v and 
                abs(dvl.bi_z_axis) < config.dvl_max_v) : 
                v = PyKDL.Vector(dvl.bi_x_axis, dvl.bi_y_axis, dvl.bi_z_axis)
                dvl_update = 1
        elif dvl.wi_status == "A" and dvl.wi_error > -32.0:
            if (abs(dvl.wi_x_axis) < config.dvl_max_v and 
                abs(dvl.wi_y_axis) < config.dvl_max_v and 
                abs(dvl.wi_z_axis) < config.dvl_max_v) : 
                v = PyKDL.Vector(dvl.wi_x_axis, dvl.wi_y_axis, dvl.wi_z_axis)
                dvl_update = 2
        
        #Filter to check if the altitude is reliable
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            config.bottom_status =  config.bottom_status + 1
        else:
            config.bottom_status = 0
        
        if config.bottom_status > 4:
            self.vehicle.altitude = dvl.bd_range
        else:
            self.vehicle.altitude = INVALID_ALTITUDE
            
        if dvl_update != 0:
            #Rotate DVL velocities and Publish
            #Compte! EL DVL no es dextrogir i s'ha de negar la Y
            vr = config.dvl_tf.M * v
            distance = config.dvl_tf_data[0:3]
            #dvl_reference = "bottom" if dvl_update == 1 else "water"
            mode = 'b' if (dvl_update == 1) else 'w'
            #if self.__LOCK__.locked():
            #    self.ros.NO_LOCK_ACQUIRE += 1
            #    return
            if dvl.header.stamp < config.last_time.predict:
                self.ros.NO_LOCK_ACQUIRE += 1
                return
            self.__LOCK__.acquire()
            try:
                if dvl.header.stamp < config.last_time.predict:
                    self.ros.NO_LOCK_ACQUIRE += 1
                    return
                self.vehicle.twist_linear = np.array([vr[0], -vr[1], vr[2]])
                # Ara ja tenim la velocitat lineal en el DVL representada en 
                # eixos de vehicle falta calcular la velocitat lineal al 
                # centre del vehicle en eixos del vehicle
                angular_velocity = self.vehicle.twist_angular
                self.vehicle.twist_linear -= np.cross(angular_velocity, 
                                                      distance)
                self.predict(config.last_time.dvl)
                
                self.slam_worker.update_dvl(self.vehicle.twist_linear, mode)
                self.ros.last_update_time = config.last_time.dvl
                self.slam_worker.resample()
            except:
                exc_info = sys.exc_info()
                print "DVL UPDATE ERROR:\n", traceback.print_tb(exc_info[2])
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', 
                          self.ros.name)
        
    
    def update_svs(self, svs):
        #print os.getpid()
        config = self.config
        config.last_time.svs = copy.copy(svs.header.stamp)
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
        vehicle_rpy = PyKDL.Rotation.RPY(*pose_angle)
        svs_trans = config.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data - svs_trans
        
        if not config.init.svs:
            config.init.svs = True
            self.__LOCK__.acquire()
            try:
                print "SVS: INITIALISING DEPTH to ", str(svs_data[2])
                self.vehicle.pose_position[2] = svs_data[2]
                self.slam_worker.vehicle.states[:, 2] = svs_data[2]
            except:
                exc_info = sys.exc_info()
                print "G500_SLAM:UPDATE_SVS():\n", traceback.print_tb(exc_info[2])
            finally:
                self.__LOCK__.release()
            return
        
        if self.__LOCK__.locked():
            return
        self.__LOCK__.acquire()
        try:
            self.vehicle.pose_position[2] = svs_data[2]
            if self.predict(config.last_time.svs):
                #self.slam_worker.update_svs(self.vehicle.pose_position[2])
                self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
                self.ros.last_update_time = config.last_time.svs
        except:
            exc_info = sys.exc_info()
            print "G500_SLAM:UPDATE_SVS():\n", traceback.print_tb(exc_info[2])
        finally:
            self.__LOCK__.release()
        #self.publish_data()

    
    def update_imu(self, imu):
        #print os.getpid()
        ret_val = None
        config = self.config
        config.init.imu = True
        
        pose_angle = tf.transformations.euler_from_quaternion(
                                       [imu.orientation.x, imu.orientation.y, 
                                       imu.orientation.z, imu.orientation.w])
        imu_data =  PyKDL.Rotation.RPY(*pose_angle)
        imu_data = imu_data*config.imu_tf.M
        pose_angle = imu_data.GetRPY()
        if not config.imu_data :
            config.last_imu_orientation = pose_angle
            config.last_time.imu = copy.copy(imu.header.stamp)
            #config.imu_data = True
            # Initialize heading buffer to apply a savitzky_golay derivation
            if len(config.heading_buffer) == 0:
                config.heading_buffer.append(pose_angle[2])
                
            inc = normalize_angle(pose_angle[2] - config.heading_buffer[-1])
            config.heading_buffer.append(config.heading_buffer[-1] + inc)
            
            if len(config.heading_buffer) == len(config.savitzky_golay_coeffs):
                print "IMU initialised!"
                config.imu_data = True
            
        else:
            period = (imu.header.stamp - config.last_time.imu).to_sec()
            pose_angle_quaternion = \
                tf.transformations.quaternion_from_euler(*pose_angle)
            config.last_imu_orientation = pose_angle
            self.__LOCK__.acquire()
            try:
                self.vehicle.pose_orientation = pose_angle_quaternion
                
                # Derive angular velocities from orientations
                self.vehicle.twist_angular = \
                    normalize_angle(np.array(pose_angle)- 
                        np.array(config.last_imu_orientation))/period
                
                # For yaw rate we apply a savitzky_golay derivation
                inc = normalize_angle(pose_angle[2] - 
                                      config.heading_buffer[-1])
                config.heading_buffer.append(config.heading_buffer[-1] + inc)
                config.heading_buffer.pop(0)
                self.vehicle.twist_angular[2] = \
                    np.convolve(config.heading_buffer, 
                                config.savitzky_golay_coeffs, 
                                mode='valid') / period
                config.last_time.imu = copy.copy(imu.header.stamp)
                
                self.predict(imu.header.stamp)
                self.ros.last_update_time = imu.header.stamp
                ###############################################################
            except:
                exc_info = sys.exc_info()
                print "G500_SLAM:UPDATE_IMU():\n", traceback.print_tb(exc_info[2])
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        return ret_val
        
        
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
            if eff_nparticles < 0:
                # Shift the states so that the central state is equal to the mean
                state_est = self.slam_worker.estimate()
                state_delta = (self.slam_worker.vehicle.states[0, 0:3] - 
                               state_est.vehicle.ned.state)
                self.slam_worker.vehicle.states[:, 0:3] += state_delta
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
        
    def slam_housekeeping(self, *args, **kwargs):
        self.__LOCK__.acquire()
        try:
            self.slam_worker.compress_maps()
        except:
            exc_info = sys.exc_info()
            print "G500_SLAM:SLAM_HOUSEKEEPING():\n", traceback.print_tb(exc_info[2])
        finally:
            self.__LOCK__.release()
        
    def predict(self, predict_to_time):
        config = self.config
        if not config.init.init:
            time_now = predict_to_time
            config.last_time.predict = copy.copy(time_now)
            self.slam_worker.last_time.predict = time_now.to_sec()
            if config.imu_data and config.gps_data:                
                # Initialise slam worker with north and east co-ordinates
                init = lambda:0
                init.north = config.init.north
                init.east = config.init.east
                self.slam_worker.reset_states()
                print "Resetting states to ", \
                    str(init.north), ", ", str(init.east)
                self.set_navigation(init)
                self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
                config.init.init = True
            return False
        else:
            pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
            if predict_to_time < config.last_time.predict:
                self.ros.NO_LOCK_ACQUIRE += 1
                return False
            elif predict_to_time == config.last_time.predict:
                return True
            time_now = predict_to_time
            config.last_time.predict = copy.copy(time_now)
            time_now = time_now.to_sec()
            self.slam_worker.predict(np.array(pose_angle), time_now)
            return True
    
    def publish_transforms(self, *args, **kwargs):
        timestamp = rospy.Time.now()
        world_frame_id = "world"
        platform_orientation = self.vehicle.pose_orientation
        br = tf.TransformBroadcaster()
        br.sendTransform(self.slam_worker.vehicle.states[0, :3], 
                         platform_orientation, timestamp, 
                         self.ros.name, world_frame_id)
        # Publish stereo_camera tf relative to each slam particle
        o_zero = tf.transformations.quaternion_from_euler(0, 0, 0, 'sxyz')
        o_st_down = tf.transformations.quaternion_from_euler(0.0, 0.0, -1.57, 'sxyz')
        o_st_front = tf.transformations.quaternion_from_euler(-1.57, 0.0, -1.57, 'sxyz')
        if "down" in CAMERA_ORIENTATION:
            o_sensor = o_st_down
        elif ("front" in CAMERA_ORIENTATION):
            o_sensor = o_st_front
        else:
            raise rospy.ROSException("Camera orientation undefined!")
        nparticles = self.slam_worker.vars.nparticles
        for particle_idx in range(nparticles):
            str_particle_idx = str(particle_idx)
            child_name = "slam_sensor" + str_particle_idx
            child_right_name = "slam_sensor_right" + str_particle_idx
            parent_name = "slam_base" + str_particle_idx
            # Publish parent particle
            parent_state = self.slam_worker.vehicle.states[particle_idx, :3]
            if not np.any(np.isnan(parent_state)):
                br.sendTransform(tuple(parent_state), platform_orientation,
                                 timestamp, parent_name, world_frame_id)
                br.sendTransform((0.0, 0.0, -0.7), o_sensor, timestamp, 
                                 child_name, parent_name)
                br.sendTransform((0.12, 0.0, 0.0), o_zero, timestamp, 
                                 child_right_name, child_name)
                """
                br.sendTransform(tuple(parent_state), platform_orientation,
                                 timestamp, parent_name, world_frame_id)
                br.sendTransform((0.0, -0.06, -0.7), zero_orientation,
                                 timestamp, child_name, parent_name)
                br.sendTransform((0.0, 0.12, 0.0), zero_orientation,
                                 timestamp, child_right_name, child_name)
                """
        
    
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
        
    def debug_print(self, *args, **kwargs):
        print "Weights: "
        #print self.slam_worker.states
        print self.slam_worker.vehicle.weights
    

def main():
    try:
        # Init node
        rospy.init_node('phdslam')
        girona500_navigator = G500_SLAM(rospy.get_name())
        if not __PROFILE__:
            rospy.spin()
        else:
            import time
            try:
                imu_msg = rospy.wait_for_message("/cola2_navigation/imu", 
                                                 Imu, 1)
                last_time = imu_msg.header.stamp
                TEST_IMU = True
            except rospy.ROSException:
                print "*** timeout waiting for imu message! ***"
                imu_msg = None
                TEST_IMU = False
            
            try:
                gps_msg = rospy.wait_for_message(
                        "/cola2_navigation/fastrax_it_500_gps", 
                        FastraxIt500Gps, 1)
                last_time = gps_msg.header.stamp
                TEST_GPS = True
            except rospy.ROSException:
                print "*** timeout waiting for gps message! ***"
                gps_msg = None
                TEST_GPS = False
            
            try:
                svs_msg = rospy.wait_for_message(
                    "/cola2_navigation/valeport_sound_velocity", 
                    ValeportSoundVelocity)
                last_time = svs_msg.header.stamp
                TEST_SVS = True
            except rospy.ROSException:
                print "*** timeout waiting for svs message! ***"
                svs_msg = None
                TEST_SVS = False
                
            try:
                dvl_msg = rospy.wait_for_message(
                    "/cola2_navigation/teledyne_explorer_dvl",
                    TeledyneExplorerDvl)
                last_time = dvl_msg.header.stamp
                TEST_DVL = True
            except rospy.ROSException:
                print "*** timeout waiting for dvl message! ***"
                dvl_msg = None
                TEST_DVL = False
                
            try:
                pcl_msg = rospy.wait_for_message(
                    "/slamsim/features", PointCloud2, 2)
                last_time = pcl_msg.header.stamp
                TEST_PCL = True
            except rospy.ROSException:
                print "*** timeout waiting for pcl message! ***"
                pcl_msg = None
                TEST_PCL = False
            
            print "Pausing for 3 seconds..."
            time.sleep(3)
            test_msg_list = (imu_msg, gps_msg, svs_msg, dvl_msg, pcl_msg)
            test_flag_list = (TEST_IMU, TEST_GPS, TEST_SVS, TEST_DVL, TEST_PCL)
            test_str_list = ("imu", "gps", "svs", "dvl", "pcl")
            test_fn_list = (girona500_navigator.update_imu,
                            girona500_navigator.update_gps, 
                            girona500_navigator.update_svs,
                            girona500_navigator.update_dvl,
                            girona500_navigator.update_features)
            for test_msg, test_flag, test_str, test_fn in \
                    zip(test_msg_list, test_flag_list, 
                        test_str_list, test_fn_list):
                if test_flag:
                    print "\n\nTesting ", test_str
                    for count in range(__PROFILE_NUM_LOOPS__):
                        last_time.secs += 1
                        test_msg.header.stamp = last_time
                        test_fn(test_msg)
                        percent = int(round((count*100.0)/__PROFILE_NUM_LOOPS__))
                        sys.stdout.write("\r%d%% complete" %percent)    # or print >> sys.stdout, "\r%d%%" %i,
                        sys.stdout.flush()
            
            print "\n** Finished profiling **\n"
            rospy.signal_shutdown("Finished profiling.")
    except rospy.ROSInterruptException: 
        pass


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        girona500_navigator = G500_SLAM(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass

