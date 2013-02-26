#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lib.misctools import STRUCT, image_converter, pcl_xyz_cov
import threading
import time
import numpy as np
import roslib
import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Int32
from offline_slam import G500_SLAM
import rosbag
import glob
import slam_feature_detector
import argparse
import tempfile
import copy
import pylab
pylab.ion()
import code
import pickle
import csv
import subprocess
import signal
import cv2
import os
import math
from subprocess import call
import yaml
from collections import namedtuple

#for plotting
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mpl.rcParams['legend.fontsize'] = 10

MIXTURE = namedtuple("MIXTURE", "weights states covs parent_ned parent_rpy")

class ROSClock(STRUCT):
    def __init__(self, init_time=None, publish_freq=1000):
        self._time_ = None
        self._RUN_ = False
        self._init_time_ = None
        self._INITIALISED_ = False
        self.set_time(init_time)
        self._sleep_time_ = 1./float(publish_freq)
        self._clock_msg_ = Clock()
        self._clock_pub_ = rospy.Publisher("/clock", Clock)
        self._secs_elapsed_ = rospy.Publisher("/clock_secs_elapsed", Int32)
        self._clock_thread_ = threading.Thread(target=self.publish_clock)
        self._clock_thread_.daemon = True
    
    def get_time(self):
        return copy.deepcopy(self._time_)
    
    def get_elapsed_time(self):
        return (self._time_ - self._init_time_).to_sec()
    
    def set_time(self, new_time):
        if new_time is None:
            return
        if type(new_time) == float:
            new_time = rospy.Time.from_sec(new_time)
        self._time_ = new_time
        if not self._INITIALISED_:
            self._init_time_ = copy.deepcopy(self._time_)
            self._INITIALISED_ = True
    
    def publish_clock(self):
        while self._RUN_:
            self._clock_msg_.clock = self._time_
            self._clock_pub_.publish(self._clock_msg_)
            secs_elapsed = (self._time_ - self._init_time_).to_sec()
            self._secs_elapsed_.publish(Int32(secs_elapsed))
            time.sleep(self._sleep_time_)
    
    def start(self):
        assert (not self._time_ is None), "Time not initialised. Use set_time()"
        self._RUN_ = True
        self._clock_thread_.start()
    
    def stop(self):
        self._RUN_ = False
    
bag_clock = ROSClock()

"""
class BagProcessor(STRUCT):
    def __init__(self, callbacks={}):
        self._callbacks_ = callbacks
    
    def add_callbacks(self, new_callback):
        self._callbacks_.update(new_callback)
    
    def do_callbacks(self, msg):
        pass
""" 

class message_loopback(object):
    def __init__(self, out_topic=None, out_type=None, in_topic=None, in_type=None):
        self._msg_in_ = None
        self._msg_out_ = None
        self._updated_ = False
        if not out_topic is None:
            self.PUBLISHER = True
            self.pub = rospy.Publisher(out_topic, out_type)
        else:
            self.PUBLISHER = False
        if not in_topic is None:
            self.SUBSCRIBER = True
            self.sub = rospy.Subscriber(in_topic, in_type, self.update_msg)
        else:
            self.SUBSCRIBER = False
        self._LOCK_ = threading.Lock()
    
    def update_msg(self, msg):
        self._LOCK_.acquire()
        try:
            self._msg_in_ = msg
            self._updated_ = True
        finally:
            self._LOCK_.release()
    
    def new_msg_available(self):
        return self._updated_
    
    def get_msg(self):
        self._LOCK_.acquire()
        try:
            self._updated_ = False
            return self._msg_in_
        finally:
            self._LOCK_.release()
    
    def get_new_msg(self, timeout=0.2):
        start_time = time.time()
        while True:
            if self._updated_:
                return self.get_msg()
            if time.time()-start_time > timeout:
                return None
            time.sleep(0.01)
    
    def publish(self, msg):
        assert self.PUBLISHER, "Trying to publish via non-publisher"
        self._msg_out_ = msg
        self.pub.publish(self._msg_out_)
    

def process_bag_file(bag_filename, g500slam, callback_dictionary):
    global bag_clock
    bag = rosbag.Bag(bag_filename)
    itr = bag.read_messages()
    for bag_entry in itr:
        bag_clock.set_time(bag_entry[2])
        msg_topic = bag_entry[0]
        msg = bag_entry[1]
        #header = getattr(msg, "header", None)
        #if not header is None:
        #    print "delta time = %s" % (msg.header.stamp-bag_clock.get_time()).to_sec()
        #    bag_clock.set_time(msg.header.stamp)
        g500slam.publish_transforms()
        if ((msg_topic in callback_dictionary) and 
            (not callback_dictionary[msg_topic] is None)):
            callback_dictionary[msg_topic](msg)
        g500slam.publish_data()

def rosinit():
    # Pick a random port and attempt to start
    roscore_started = 0
    while not roscore_started is None:
        roscore_port = np.random.randint(1025, 65535)
        roscore_process = subprocess.Popen(["roscore", "-p"+str(roscore_port)])
        print "Waiting for roscore to settle..."
        time.sleep(5)
        roscore_started = roscore_process.poll()
    # Set ros master uri
    os.environ["ROS_MASTER_URI"] = 'http://localhost:'+str(roscore_port)
    
    # Load default parameters
    
    cola2_base_dir = roslib.stacks.get_stack_dir("cola2")
    udg_pandora_base_dir = roslib.packages.get_pkg_dir("udg_pandora")
    param_files = [
        cola2_base_dir+"/cola2_safety/config/safety_g500.yaml",
        cola2_base_dir+"/cola2_safety/config/fake_main_board.yaml",
        cola2_base_dir+"/cola2_navigation/config/dynamics_odin.yaml",
        cola2_base_dir+"/cola2_control/config/thruster_allocator_sim.yaml",
        cola2_base_dir+"/cola2_control/config/velocity_controller_sim.yaml",
        cola2_base_dir+"/cola2_control/config/joy.yaml",
        cola2_base_dir+"/cola2_control/config/pilot.yaml",
        cola2_base_dir+"/cola2_navigation/config/navigator.yaml",
        cola2_base_dir+"/cola2_control/../launch/mission.yaml",
        cola2_base_dir+"/cola2_control/config/arm_controller_5DoF.yaml",
        cola2_base_dir+"/cola2_safety/../launch/basic_mission_parameters.yaml",
        udg_pandora_base_dir+"/config/slam_feature_detector.yaml",
        udg_pandora_base_dir+"/config/phdslam.yaml"]
    for _param_file_ in param_files:
        subprocess.call(["rosparam", "load", _param_file_])
    config = yaml.load(open(udg_pandora_base_dir+"/config/phdslam.yaml"))
    config.update(yaml.load(open(udg_pandora_base_dir+"/config/slam_feature_detector.yaml")))
    return roscore_port, roscore_process, config

def plotEllipse(pos, P, edge='black', face='0.3'):
    # Copyright Tinne De Laet, Oct 2009
    U, s , Vh = np.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/np.pi
    ellipsePlot = Ellipse(xy=pos, width=2.0*math.sqrt(s[0]), 
        height=2.0*math.sqrt(s[1]), angle=orient, facecolor=face, edgecolor=edge)
    ellipsePlot.set_alpha(0.6)
    ax = mpl.pyplot.gca()
    ax.add_patch(ellipsePlot);
    return ellipsePlot;

def process_bags(bagfiles_list, roscore_process, config, PERFORM_SLAM=False, OUT_DIR=None):
    image_proc_process = None
    try:
        if PERFORM_SLAM:
            print "Running simulation with SLAM"
        else:
            print "Running simulation without SLAM"
        global bag_clock
        rospy.set_param("/use_sim_time", True)
        # Extract initial time from first message of the bag file
        first_bagfile = rosbag.Bag(bagfiles_list[0])
        init_time = first_bagfile.read_messages().next()[2]
        first_bagfile.close()
        # Discard first messages up to specified time
        CLOCK_BURN_TIME = rospy.Duration(0)
        # Initialise the clock with the time and start
        bag_clock.set_time(init_time+CLOCK_BURN_TIME)
        bag_clock.start()
        
        USE_PF = PERFORM_SLAM
        # Create g500slam
        if not USE_PF:
            # Using EKF
            nparticles = 1
        else:
            # OR particle filter
            nparticles = 50
            #g500slam.slam_worker._filter_update_ = g500slam.slam_worker._pf_update_
            #g500slam.slam_worker._filter_update_ = g500slam.slam_worker._opt_pf_update_
        g500slam = G500_SLAM("phdslam", nparticles=nparticles)
        config["phdslam/nparticles"] = nparticles
        
        print "Creating callback dictionary..."
        # Create dictionary for callbacks
        SENSOR_ROOT = "/navigation_g500/" #"/cola2_navigation/"
        BAG_CAMERA_ROOT = "/stereo_down/" #"/stereo_camera/"
        PUB_CAMERA_ROOT = "/stereo_camera/" #"/stereo_camera"+tempfile.mktemp(prefix='', dir='')+'/'
        cam_msg_stamps = range(4)
        
        if PERFORM_SLAM:
            # Start visual detector - only needed for slam
            v_detector = slam_feature_detector.SlamFeatureDetector(rospy.get_name())
            
            features_topic = "/slam_feature_detector/features"
            features = message_loopback(in_topic=features_topic, in_type=PointCloud2)
            img_features = message_loopback(in_topic="/slam_feature_detector/features_img_l", in_type=Image)
            img_landmarks_pub = message_loopback("/phdslam/img_landmarks", Image)
            imc = image_converter()
        cam_img_l = message_loopback(PUB_CAMERA_ROOT+"left/image_raw", Image,
                                     PUB_CAMERA_ROOT+"left/image_rect", Image)
        cam_info_l = message_loopback(PUB_CAMERA_ROOT+"left/camera_info", CameraInfo)
        cam_img_r = message_loopback(PUB_CAMERA_ROOT+"right/image_raw", Image,
                                     PUB_CAMERA_ROOT+"right/image_rect", Image)
        cam_info_r = message_loopback(PUB_CAMERA_ROOT+"right/camera_info", CameraInfo)
        
        # Read bumblebee pickle file
        camera_pickle_file = "bumblebee.p"
        print "Loading information from "+camera_pickle_file
        camera_info_pickle = roslib.packages.find_resource("udg_pandora",
            camera_pickle_file)
        if len(camera_info_pickle):
            camera_info_pickle = camera_info_pickle[0]
            try:
                cinfo_l, cinfo_r = tuple(
                    pickle.load(open(camera_info_pickle, "rb")))
            except IOError:
                print "Failed to load camera information!"
                rospy.logfatal("Could not read camera parameters")
                raise rospy.exceptions.ROSException(
                    "Could not read camera parameters")
        def pub_cinfo_l(msg):
            print "pub_cinfo_l"
            #code.interact(local=locals())
            cinfo_l.header = copy.copy(msg.header)
            cam_info_l.publish(cinfo_l)
        def pub_cinfo_r(msg):
            print "pub_cinfo_r"
            #code.interact(local=locals())
            cinfo_r.header = copy.copy(msg.header)
            cam_info_r.publish(cinfo_r)
        pub_cinfo_l.cinfo_l = cinfo_l
        pub_cinfo_r.cinfo_r = cinfo_r
        
        callback_dictionary = {
            SENSOR_ROOT+"imu"                       : g500slam.update_imu, 
            SENSOR_ROOT+"teledyne_explorer_dvl"     : g500slam.update_dvl,
            SENSOR_ROOT+"linkquest_navquest600_dvl" : g500slam.updateLinkquestDvl,
            SENSOR_ROOT+"tritech_igc_gyro"          : None,
            SENSOR_ROOT+"valeport_sound_velocity"   : g500slam.update_svs}
        #if PERFORM_SLAM:
        callback_dictionary.update({
                BAG_CAMERA_ROOT+"left/image_raw"    : cam_img_l.publish,
                BAG_CAMERA_ROOT+"left/camera_info"  : cam_info_l.publish,
                BAG_CAMERA_ROOT+"right/image_raw"   : cam_img_r.publish,
                BAG_CAMERA_ROOT+"right/camera_info" : cam_info_r.publish,
                "/slam_feature_detector/features"   : g500slam.update_features})
        
        # File name for output bag
        if OUT_DIR is None:
            datetimestr = time.strftime("%d%m%Y_%H%M%S", time.localtime())
            if PERFORM_SLAM:
                dir_prefix = "slamout_withslam_"+datetimestr+"_"
            else:
                dir_prefix = "slamout_"+datetimestr+"_"
        else:
            dir_prefix=""
        
        print "Completed initialisation"
        str_input = raw_input("""Continue? Type "yes" to continue: """)
        if not str_input == "yes":
            print "Aborting!"
            return
        
        if PERFORM_SLAM:
            image_proc_cmd = ['rosrun', 'stereo_image_proc', 'stereo_image_proc','__ns:='+PUB_CAMERA_ROOT[:-1]]
            print '=== running stereo_img_proc:',' '.join(image_proc_cmd)
            image_proc_process = subprocess.Popen(image_proc_cmd)
            # image viewers:
            # image rect
            subprocess.Popen(["rosrun", "image_view", "image_view", "image:=/stereo_camera/left/image_rect_color"])
            # features
            subprocess.Popen(["rosrun", "image_view", "image_view", "image:=/slam_feature_detector/features_img_l"])
            # image landmarks
            subprocess.Popen(["rosrun", "image_view", "image_view", "image:=/phdslam/img_landmarks"])
        
        # Store value to plot
        timestamp = [init_time.to_sec()]
        nav_ned = [(0., 0., 0.)]
        weights = []
        plot_last_time = bag_clock.get_elapsed_time()
        pcl_helper = pcl_xyz_cov()
        
        # Open output bag file
        if OUT_DIR is None:
            out_directory = tempfile.mkdtemp(prefix=dir_prefix, dir='./')
        else:
            out_directory = OUT_DIR
            call(["mkdir", "-p", OUT_DIR])
        yaml.dump(config, open(out_directory+"/config.yaml", 'w'))
        last_transform_time = rospy.Time(0)
        slam_cam_init = False
        img_savefile_num = 0
        update_figures = False
        sync_cam_msgs = False
        
        for bagfile in bagfiles_list:
            print "Processing ", bagfile
            out_bagfile = out_directory+"/"+bagfile[bagfile.rfind('/')+1:]
            slamout_bag = rosbag.Bag(out_bagfile, "w", compression="bz2")
            
            bag = rosbag.Bag(bagfile)
            itr = bag.read_messages()
            for bag_entry in itr:
                if (bag_entry[2]-bag_clock.get_time()).to_sec() > 0:
                    bag_clock.set_time(bag_entry[2])
                msg_topic = bag_entry[0]
                msg = bag_entry[1]
                #header = getattr(msg, "header", None)
                #if not header is None:
                #    print "delta time = %s" % (msg.header.stamp-bag_clock.get_time()).to_sec()
                #    bag_clock.set_time(msg.header.stamp)
                
                if ((msg_topic in callback_dictionary) and 
                    (not callback_dictionary[msg_topic] is None) and 
                    (bag_entry[2] >= init_time)):
                    
                    #if PERFORM_SLAM and (bag_entry[2] - last_transform_time).to_sec() >= 0.01:
                    #    g500slam.publish_transforms()
                    #    last_transform_time = bag_entry[2]
                    
                    #print "Calling: ", callback_dictionary[msg_topic]
                    callback_dictionary[msg_topic](msg)
                    if msg_topic == BAG_CAMERA_ROOT+"left/image_raw":
                        cam_msg_stamps[0] = msg.header.stamp
                    elif msg_topic == BAG_CAMERA_ROOT+"left/camera_info":
                        cam_msg_stamps[1] = msg.header.stamp
                        caminfo_msg_l = msg
                    elif msg_topic == BAG_CAMERA_ROOT+"right/image_raw":
                        cam_msg_stamps[2] = msg.header.stamp
                    elif msg_topic == BAG_CAMERA_ROOT+"right/camera_info":
                        cam_msg_stamps[3] = msg.header.stamp
                        caminfo_msg_r = msg
                    
                    if (cam_msg_stamps[0] == cam_msg_stamps[1] == cam_msg_stamps[2] == cam_msg_stamps[3]):
                        sync_cam_msgs = True
                        update_figures = True
                        cam_msg_stamps = range(4)
                    else:
                        sync_cam_msgs = False
                    
                    if PERFORM_SLAM and sync_cam_msgs:
                        if not slam_cam_init:
                            #print "Initialising camera info from bag"
                            #g500slam.fromCameraInfo(caminfo_msg_l, caminfo_msg_r)
                            #v_detector.fromCameraInfo(caminfo_msg_l, caminfo_msg_r)
                            slam_cam_init = True
                        g500slam.publish_transforms()
                        print "waiting for features"
                        msg = features.get_new_msg()
                        if not msg is None:
                            msg.header.stamp = rospy.Time(0)
                            #print "updating with features "
                            callback_dictionary[features_topic](msg)
                            
                            # Get the features_img
                            features_img_l = img_features.get_new_msg()
                            if not features_img_l is None:
                                cvim = np.asarray(imc.cvimage(features_img_l))
                            else:
                                cvim = np.asarray(imc.cvimage(cam_img_l.get_msg()))
                            
                            # Get image landmarks
                            if len(g500slam.image_landmarks):
                                img_landmarks = g500slam.image_landmarks[0].astype(np.int)
                            else:
                                img_landmarks = np.zeros(0)
                            if img_landmarks.shape[0]:
                                print "Image landmarks:\n", img_landmarks
                                #cvim[img_landmarks[:, 1], img_landmarks[:, 0]] = 255
                                #cvim[img_landmarks[:, 1]+1, img_landmarks[:, 0]] = 0
                                #cvim[img_landmarks[:, 1], img_landmarks[:, 0]+1] = 0
                                #cvim[img_landmarks[:, 1]+1, img_landmarks[:, 0]+1] = 255
                                for landmark in img_landmarks:
                                    cv2.circle(cvim, tuple(landmark), 5, (255, 255, 255), 2)
                            
                            img_landmarks_msg = imc.img_msg(cv2.cv.fromarray(cvim))
                            img_landmarks_pub.publish(img_landmarks_msg)
                            
                    # Publish and save nav_sts
                    nav_msg, pcl_map_msg, cam_nav_msg, mixture = g500slam.publish_data()
                    
                    if not nav_msg is None:
                        slamout_bag.write("/phdslam/nav_sts", nav_msg, bag_entry[2])
                        slamout_bag.write("/phdslam/features", pcl_map_msg, bag_entry[2])
                        try:
                            slamout_bag.write("/phdslam/cam_nav_sts", cam_nav_msg, bag_entry[2])
                        except:
                            code.interact(local=locals())
                        if (nav_msg.header.stamp.to_sec()-timestamp[-1]) > 0.25:
                            timestamp.append(nav_msg.header.stamp.to_sec())
                            nav_ned.append((nav_msg.position.north,
                                            nav_msg.position.east,
                                            nav_msg.position.depth))
                            weights = g500slam.slam_worker.vehicle.weights
                if update_figures and not ((nav_msg is None) or (pcl_map_msg is None)): #bag_clock.get_elapsed_time()-plot_last_time > 0.5:
                    update_figures = False
                    filename_num_str = "%06.0f" % img_savefile_num
                    
                    if not mixture is None:
                        mixture_pickle_filename = out_directory+"/mixture"+filename_num_str+".p"
                        p_obj = MIXTURE(mixture.weights, mixture.states, 
                                           mixture.covs, mixture.parent_ned, mixture.parent_rpy)
                        pickle_file = open(mixture_pickle_filename, "w")
                        try:
                            pickle.dump(p_obj, pickle_file)
                            pickle_file.close()
                        except:
                            print "Pickling error"
                            code.interact(local=locals())
                        #mixture.save_to_file(mixture_pickle_filename)
                        
                    # trajectory
                    fig = plt.figure(1)
                    ax = fig.gca()
                    ax.cla()
                    ned = np.asarray(nav_ned)
                    ax.plot(ned[:, 0], ned[:, 1], label="Estimated trajectory")
                    #plot_limits = [-1, 4, -7, 3]
                    plot_limits = [-3.5, 3.5, -3.5, 3.5]
                    ax.axis(plot_limits)
                    ax.grid(True)
                    ax.legend()
                    figure_filename = out_directory+"/trajectory_"+filename_num_str+".png"
                    fig.savefig(figure_filename)
                    
                    landmarks = pcl_helper.from_pcl(pcl_map_msg)
                    if landmarks.shape[0]:
                        ax.scatter(landmarks[:, 0], landmarks[:, 1], label="Landmarks")
                    ax.axis(plot_limits)
                    ax.grid(True)
                    ax.legend()
                    filename_num_str = "%06.0f" % img_savefile_num
                    figure_filename = out_directory+"/estimate_"+filename_num_str+".png"
                    fig.savefig(figure_filename)
                    
                    # weights
                    pylab.figure(2)
                    ax = mpl.pyplot.gca()
                    ax.cla()
                    pylab.barh(np.arange(len(weights))+0.5, weights, align="center")
                    
                    # scattered particles/covariance
                    fig = plt.figure(3)
                    ax = fig.gca()
                    ax.cla()
                    try:
                        plotEllipse(g500slam.slam_worker.vehicle.kf_state[:2],
                            g500slam.slam_worker.vehicle.kf_state_cov[:2, :2])
                    except AttributeError:
                        pass
                    if USE_PF:
                        particles_xyz = g500slam.slam_worker.get_position()
                        ax.scatter(particles_xyz[:, 0],
                                   particles_xyz[:, 1])
                        try:
                            plotEllipse(g500slam.slam_worker.vehicle.kf_state[:2],
                                        g500slam.slam_worker.vehicle.kf_state_cov[:2, :2])
                        except AttributeError:
                            pass
                    else:
                        try:
                            plotEllipse(g500slam.slam_worker.vehicle.states[0, :2],
                                    g500slam.slam_worker.vehicle.covs[0])
                        except AttributeError:
                            print "Error getting covariance ellipse"
                            
                    ax.axis(plot_limits)
                    figure_filename = out_directory+"/particles_"+filename_num_str+".png"
                    fig.savefig(figure_filename)
                    pylab.draw()
                    pylab.show()
                    
                    img_savefile_num += 1
                    #if img_savefile_num >= 10:
                    #    return
                    plot_last_time = bag_clock.get_elapsed_time()
                # We can't quit on Ctrl-C, so watch for roscore instead
                if not roscore_process.poll() is None:
                    bag.close()
                    slamout_bag.close()
                    return
            bag.close()
            slamout_bag.close()
        print "Saved filtered navigation to ", out_directory
        
        # Save figure last - in case it causes an exception
        fig = pylab.figure(1)
        figure_png = out_bagfile.rstrip("bag")+"png"
        fig.savefig(figure_png)
        figure_svg = out_bagfile.rstrip("bag")+"svg"
        fig.savefig(figure_svg)
    finally:
        # stop processes when playback finished
        if not image_proc_process is None:
            image_proc_process.send_signal(signal.SIGINT)
            image_proc_process.wait()
    
def plot_bag_map(bagfilename):
    # Open the bag file
    bagfile = rosbag.Bag(bagfilename)
    itr = bagfile.read_messages(["/phdslam/features"])
    for bag_entry in itr:
        msg = bag_entry[1]
    timestamps = np.float(msg.header.stamp.to_sec())
    pcl_helper = pcl_xyz_cov()
    landmarks = pcl_helper.from_pcl(msg)
    
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(landmarks[:, 0], landmarks[:, 1], label='Landmarks')
    ax.legend()
    plt.show()
    return timestamps, landmarks

def plot_bag_result(bagfilename, PLOT3D=False, PLOT=True):
    # Open the bag file
    bagfile = rosbag.Bag(bagfilename)
    itr = bagfile.read_messages(["/phdslam/nav_sts"])
    timestamps = []
    position = []
    for bag_entry in itr:
        msg = bag_entry[1]
        timestamps.append(msg.header.stamp.to_sec())
        position.append((msg.position.north,
                         msg.position.east,
                         msg.position.depth))
    timestamps = np.asarray(map(np.float, timestamps))
    position = np.asarray(position)
    if PLOT:
        fig = plt.figure()
        if PLOT3D:
            ax = fig.gca(projection='3d')
            ax.plot(position[:, 0], position[:, 1], position[:, 2], label='Trajectory')
        else:
            ax = fig.gca()
            ax.plot(position[:, 0], position[:, 1], label='Trajectory')
        ax.legend()
        plt.show()
    return timestamps, position

def extract_bag_results(bagfilelist, root_dir=""):
    if type(bagfilelist) is str:
        bagfilelist = [bagfilelist]
    timestamps = np.zeros(0)
    positions = np.zeros((0, 3))
    for bagfilename in bagfilelist:
        _timestamps_, _positions_ = plot_bag_result(root_dir+bagfilename, PLOT=False)
        timestamps = np.hstack((timestamps, _timestamps_))
        positions = np.vstack((positions, _positions_))
    return timestamps, positions

def plot_csv_result(csv_filename, PLOT3D=False):
    csvfile = open(csv_filename)
    reader = csv.DictReader(csvfile)
    timestamps = []
    position = []
    for msg in reader:
        timestamps.append(msg["field.header.stamp"])
        position.append((msg["field.pose.pose.position.x"],
                         msg["field.pose.pose.position.y"],
                         msg["field.pose.pose.position.z"]))
    timestamps = np.asarray(map(np.float, timestamps))
    position = np.asarray(position, dtype=np.float64)
    fig = plt.figure()
    if PLOT3D:
        ax = fig.gca(projection='3d')
        ax.plot(position[:, 0], position[:, 1], position[:, 2], label='Trajectory')
    else:
        ax = fig.gca()
        ax.plot(position[:, 0], position[:, 1], label='Trajectory')
    ax.legend()
    plt.show()
    return timestamps, position

def plot_trajectory_diff(timestamps0, timestamps1, trajectory0, trajectory1):
    timestamps = []
    traj_diff = []
    for (_time0_, _traj0_) in zip(timestamps0, trajectory0):
        idx = np.where(timestamps1 == _time0_)[0]
        if idx.shape[0]:
            idx = idx[-1]
            timestamps.append(_time0_)
            traj_diff.append(_traj0_ - trajectory1[idx])
    timestamps = np.asarray(timestamps)
    traj_diff = np.asarray(traj_diff)
    abs_diff = ((traj_diff**2).sum(axis=1))**0.5
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(timestamps, abs_diff, label='Difference in estimated position')
    ax.legend()
    plt.show()
    return timestamps, traj_diff, abs_diff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--withslam", help="process file using SLAM", action="store_true")
    parser.add_argument("--outdir")
    parser.add_argument("bagfile", nargs='+', help='bag files to process (will be sorted)')
    args = parser.parse_args()
    
    # Start roscore
    roscore_port, roscore_process, config = rosinit()
    
    rospy.init_node("phdslam")
    rospy.set_param("/use_sim_time", True)
    #bag_root = "/opt/ros/bags/20120900-bagfiles-CIRS-UIB/bags/"
    #bagfiles_list = glob.glob(bag_root+"/*.bag")
    bagfiles_list = copy.deepcopy(args.bagfile)
    bagfiles_list.sort()
    print "Received list of %s files" % len(bagfiles_list)
    for _filename_ in bagfiles_list:
        print _filename_
    try:
        process_bags(bagfiles_list, roscore_process, config, args.withslam, OUT_DIR=args.outdir)
    except rospy.ROSException:
        roscore_process.send_signal(signal.SIGINT)
        roscore_process.wait()
