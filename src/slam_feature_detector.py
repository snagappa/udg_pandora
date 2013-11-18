#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import Image, PointCloud2
import hwu_meta_data.metaclient as metaclient
import cv2
import code
import numpy as np
from lib.misctools import STRUCT, pcl_xyz_cov, approximate_mahalanobis, \
    merge_states, image_converter, pcl_xyz, camera_buffer
from lib import cameramodels, image_feature_extractor


def get_default_parameters():
    parameters = {}
    parameters["slam_feature_detector/fov/near"] = 0.15
    parameters["slam_feature_detector/fov/far"] = 4.0
    parameters["slam_feature_detector/extractor_name"] =  "Surf"
    parameters["slam_feature_detector/grid_adapted"] = False
    parameters["slam_feature_detector/num_features"] =  40
    parameters["slam_feature_detector/flann_ratio"] =  0.6
    
    # Options for the surf extractor
    parameters["slam_feature_detector/surf_extractor/hessian_threshold"] =  50
    
    # Options for the orb extractor
    #slam_feature_detector/orb_extractor/
    
    # Update rate
    parameters["slam_feature_detector/rate"] =  -1
    
    # CAMERA POSE
    # stereo_down position    : [0.6, 0.0, 0.4]
    # stereo_down orientation : [0.0, 0.0, 1.57]
    # stereo_front position   : [0.6, 0.0, 0.2]
    # stereo_front orientation: [1.57, 0.0, 1.57]
    # stereo_baseline: [0.12, 0.0, 0.0]
    parameters["slam_feature_detector/camera_position"] =  [0.6, 0.0, 0.4]
    parameters["slam_feature_detector/camera_orientation"] =  [0.0, 0.0, 1.57]
    parameters["slam_feature_detector/camera_baseline"] =  [0.12, 0.0, 0.0]
    parameters["slam_feature_detector/camera_baseline_orientation"] =  [0., 0., 0.]
    
    # use camera_root/{left,right}/image_topic
    parameters["slam_feature_detector/camera_root"] =  "/stereo_camera"
    parameters["slam_feature_detector/image_topic"] =  "image_rect"
    
    # Noise scale factors
    parameters["slam_feature_detector/sigma_scale_factor/x"] =  0.08
    parameters["slam_feature_detector/sigma_scale_factor/y"] =  0.08
    parameters["slam_feature_detector/sigma_scale_factor/z"] =  0.08
    return parameters

def set_default_parameters():
    parameters = get_default_parameters()
    for _param_name_ in parameters:
        rospy.set_param(_param_name_, parameters[_param_name_])


def merge_points(weights, states, covs, merge_threshold=1.):
    num_remaining_components = weights.shape[0]
    merged_wts = []
    merged_sts = []
    merged_cvs = []
    while num_remaining_components:
        max_wt_index = weights.argmax()
        max_wt_state = states[np.newaxis, max_wt_index]
        max_wt_cov = covs[np.newaxis, max_wt_index]
        mahalanobis_fn = approximate_mahalanobis
        mahalanobis_dist = mahalanobis_fn(max_wt_state, max_wt_cov,
                                          states)
        merge_list_indices = ( np.where(mahalanobis_dist <=
                                            merge_threshold)[0] )
        retain_idx = ( np.where(mahalanobis_dist >
                                            merge_threshold)[0] )
        new_wt, new_st, new_cv = merge_states(
                                        weights[merge_list_indices],
                                        states[merge_list_indices],
                                        covs[merge_list_indices])
        merged_wts += [new_wt]
        merged_sts += [new_st]
        merged_cvs += [new_cv]
        # Remove merged states from the list
        #retain_idx = misctools.gen_retain_idx(self.weights.shape[0],
        #                                      merge_list_indices)
        weights = weights[retain_idx]
        states = states[retain_idx]
        covs = covs[retain_idx]
        num_remaining_components = weights.shape[0]
    
    return np.array(merged_wts), np.array(merged_sts), np.array(merged_cvs)


class SlamFeatureDetector(object):
    def __init__(self, name):
        self.name = name
        # ROS image message to cvimage convertor
        self.ros2cvimg = image_converter()
        
        camera_root = rospy.get_param("slam_feature_detector/camera_root")
        image_topic = rospy.get_param("slam_feature_detector/image_topic")
        # Initialise image buffer
        self.image_buffer = camera_buffer(camera_root, image_topic, True)
        
        # Initialise feature detector for SLAM
        self.slam_features = STRUCT()
        self.init_slam_feature_detector()
        print "Completed initialisation"
    
    def init_slam_feature_detector(self):
        slam_features = self.slam_features
        # Extract parameters
        extractor_name = rospy.get_param("slam_feature_detector/extractor_name")
        update_rate = rospy.get_param("slam_feature_detector/rate")
        make_grid_adapted = rospy.get_param("slam_feature_detector/grid_adapted")
        num_features = rospy.get_param("slam_feature_detector/num_features")
        flann_ratio = rospy.get_param("slam_feature_detector/flann_ratio")
        fov_near = rospy.get_param("slam_feature_detector/fov/near")
        fov_far = rospy.get_param("slam_feature_detector/fov/far")
        x_std_base = rospy.get_param("slam_feature_detector/sigma_scale_factor/x")
        y_std_base = rospy.get_param("slam_feature_detector/sigma_scale_factor/y")
        z_std_base = rospy.get_param("slam_feature_detector/sigma_scale_factor/z")
        pixel_std = rospy.get_param("slam_feature_detector/pixel_std")
        
        # Set update rate
        update_rate = update_rate if update_rate > 0 else None
        slam_features.update_rate = update_rate
        # Set flann ratio
        slam_features.flann_ratio = flann_ratio
        # Near and far FoV
        slam_features.fov_near = fov_near
        slam_features.fov_far = fov_far
        # Noise variance scaling
        slam_features.cov_scaling = np.asarray(
            [x_std_base**2, y_std_base**2, z_std_base**2])
        slam_features.pixel_std = pixel_std
        
        # Initialise the detector and reset the number of features
        feature_extractor = getattr(image_feature_extractor, extractor_name)
        slam_features.camera = cameramodels.StereoCameraFeatureDetector(
            feature_extractor=feature_extractor, GRID_ADAPTED=False)
        
        # Set SURF parameters
        if feature_extractor is image_feature_extractor.Surf:
            # Get hessian threshold
            hessian_threshold = rospy.get_param(
                "slam_feature_detector/surf_extractor/hessian_threshold", -1)
            if hessian_threshold > 0:
                print "Setting hessian threhosld to %s" % hessian_threshold
                slam_features.camera._featuredetector_.set_hessian_threshold(hessian_threshold)
            # Set number of octaves
            n_octaves = rospy.get_param(
                "slam_feature_detector/surf_extractor/n_octaves", -1)
            if n_octaves > 0:
                slam_features.camera._featuredetector_.set_nOctaves(n_octaves)
            # Convert to grid adapted to set num_features
            #slam_features.camera._featuredetector_.make_grid_adapted()
        
        # Convert to grid adapted to set num_features
        if make_grid_adapted:
            slam_features.camera._featuredetector_.make_grid_adapted()
        
        # Set number of features
        if num_features:
            try:
                slam_features.camera._featuredetector_.set_num_features(num_features)
            except UnboundLocalError:
                print "Could not set number of features for the detector."
        
        # Extract camera info
        slam_features.camera.fromCameraInfo(*self.image_buffer.get_camera_info())
        # Set near and far field of view
        slam_features.camera.set_near_far_fov(fov_near, fov_far)
        # Create Publisher
        slam_features.pub = rospy.Publisher("/slam_feature_detector/features",
                                            PointCloud2)
        slam_features.pub2 = rospy.Publisher(
            "/slam_feature_detector/features_xyz", PointCloud2)
        slam_features.pub3 = rospy.Publisher("/slam_feature_detector/disparity", PointCloud2)
        slam_features.pcl_helper = pcl_xyz_cov()
        slam_features.pcl_helper2 = pcl_xyz()
        
        # Publish image with detected keypoints
        self.slam_features.img_pub = [
            metaclient.Publisher('/slam_feature_detector/features_img_l', Image, {}),
            metaclient.Publisher('/slam_feature_detector/features_img_r', Image, {})]
        
        # Register the callback
        slam_features.callback_id = (
            self.image_buffer.register_callback(self.detect_slam_features,
                                                slam_features.update_rate))
    
    def detect_slam_features(self, *args):
        #time_now = args[0].header.stamp
        time_now = args[0].header.stamp #rospy.Time.now()
        slam_features = self.slam_features
        cvimage = [np.asarray(self.ros2cvimg.cvimagegray(_img_)).copy() for _img_ in args]
        #cvimage = [_img_ for _img_ in args]
        points3d, (pts_l, pts_r), (kp_l, kp_r), (desc_l, desc_r) = (
            self.slam_features.camera.points3d_from_img(*cvimage,
                ratio_threshold=slam_features.flann_ratio,
                image_margins=(64, 64, 64, 64)))
        if points3d.shape[0]:
            points_range = np.sqrt((points3d**2).sum(axis=1))
        else:
            #print "no features found"
            points_range = np.empty(0, dtype=np.int32)
            pts_l = np.empty(0)
            pts_r = np.empty(0)
        #print "Found %s features at stage 1" % points3d.shape[0]
        #code.interact(local=locals())
        #print "Average distance = %s" % np.mean(points3d, axis=0)
        valid_pts_index = points_range <= slam_features.fov_far
        points3d = points3d[valid_pts_index]
        points_range = points_range[valid_pts_index]
        
        pts_l = pts_l[valid_pts_index]
        pts_r = pts_r[valid_pts_index]
        if pts_l.shape[0]:
            disparity_pts = np.hstack((pts_l, (pts_l[:, 0] - pts_r[:, 0])[:, np.newaxis]))
            disparity_covs = slam_features.pixel_std**2*np.ones((disparity_pts.shape[0], 3))
            disparity_pts_cov = np.hstack((disparity_pts, disparity_covs))
        else:
            disparity_pts_cov = np.empty(0)
        #print "Ignoring features beyond %s m; remaining = %s" % (FOV_FAR, points3d.shape[0])
        # Merge points which are close together
        weights = np.ones(points3d.shape[0])
        avg_range = np.ones(weights.shape[0])*points_range.mean()
        points3d_scale = np.hstack((avg_range[:, np.newaxis], 
                                    avg_range[:, np.newaxis], 
                                    avg_range[:, np.newaxis]))
        covs = slam_features.cov_scaling*points3d_scale[:, np.newaxis, :]*np.eye(3)[np.newaxis]
        _wts_, points3d_states, points3d_covs = merge_points(weights, points3d, covs, merge_threshold=0.05)
        print "Detected %s (%s) features" % (points3d_states.shape[0], points3d.shape[0])
        # Convert the points to a pcl message
        if points3d_covs.shape[0]:
            points3d_covs = points3d_covs[:, range(3), range(3)]
        points3d = np.hstack((points3d_states, points3d_covs))
        pcl_msg = self.slam_features.pcl_helper.to_pcl(points3d)
        pcl_msg.header.stamp = time_now
        # set the frame id from the camera
        pcl_msg.header.frame_id = self.slam_features.camera.tfFrame
        # Publish the pcl message
        self.slam_features.pub.publish(pcl_msg)
        
        pcl_msg2 = self.slam_features.pcl_helper2.to_pcl(points3d_states)
        pcl_msg2.header.stamp = time_now
        pcl_msg2.header.frame_id = "/world"
        self.slam_features.pub2.publish(pcl_msg2)
        
        pcl_msg3 = self.slam_features.pcl_helper.to_pcl(disparity_pts_cov)
        pcl_msg3.header.stamp = time_now
        pcl_msg3.header.frame_id = self.slam_features.camera.tfFrame
        self.slam_features.pub3.publish(pcl_msg3)
        print "Published ", str(disparity_pts_cov.shape[0]), " disparity points"
        
        pts_l = pts_l.astype(np.int32)
        pts_r = pts_r.astype(np.int32)
        (pts_l_reproj, pts_r_reproj) = self.slam_features.camera.project3dToPixel(points3d_states)
        pts_l_reproj = pts_l_reproj.astype(np.int)
        pts_r_reproj = pts_r_reproj.astype(np.int)
        if pts_l.shape[0]:
            cvimage[0][pts_l[:, 1], pts_l[:, 0]] = 255
            for landmark in pts_l_reproj:
                cv2.circle(cvimage[0], tuple(landmark), 5, (0, 0, 0), -1)
        if pts_r.shape[0]:
            cvimage[1][pts_r[:, 1], pts_r[:, 0]] = 255
            for landmark in pts_r_reproj:
                cv2.circle(cvimage[1], tuple(landmark), 5, (0, 0, 0), -1)
        img_msg = [self.ros2cvimg.img_msg(cv2.cv.fromarray(_scene_))
                   for _scene_ in cvimage]
        for idx in range(len(img_msg)):
            img_msg[idx].header.stamp = time_now
            self.slam_features.img_pub[idx].publish(img_msg[idx])
        
        #if rospy.Time.now().to_sec() > 1336386861.7:
        #    import matplotlib as mpl
        #    from mpl_toolkits.mplot3d import Axes3D
        #    import matplotlib.pyplot as plt
        #    import pylab
        #    pylab.ion()
        #    fig = plt.figure()
        #    ax = fig.gca(projection="3d")
        #    ax.scatter3D(points3d_states[:, 0], points3d_states[:, 1], points3d_states[:, 2])
        #    plt.draw()
        #    code.interact(local=locals())


if __name__ == '__main__':
    try:
        import subprocess
        # Load ROS parameters
        udg_pandora_base_dir = roslib.packages.get_pkg_dir("udg_pandora")
        config_file = udg_pandora_base_dir+"/config/slam_feature_detector.yaml"
        try:
            subprocess.call(["rosparam", "load", config_file])
        except IOError:
            print "Could not locate slam_feature_detector.yaml, using default parameters"
            set_default_parameters()
        im_l = np.asarray(cv2.imread("image2_rect_left.jpg"))
        im_r = np.asarray(cv2.imread("image2_rect_right.jpg"))
        rospy.init_node('slam_feature_detector')
        slam_feature_detector = SlamFeatureDetector(rospy.get_name())
        #slam_feature_detector.detect_slam_features(im_l, im_r)
        rospy.spin()
    except rospy.ROSInterruptException: pass
