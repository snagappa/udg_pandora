#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from sensor_msgs.msg import Image
#import std_msgs.msg
import std_srvs.srv
#from auv_msgs.msg import NavSts
from udg_pandora.msg import Detection
import hwu_meta_data.metaclient as metaclient
import cv2
import code
from lib import objdetect
import numpy as np

from lib.misctools import STRUCT, image_converter, camera_buffer #, Retinex
from lib import image_feature_extractor
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion, \
    quaternion_matrix, euler_matrix
from geometry_msgs.msg import PoseWithCovarianceStamped
import itertools
from lib.cameramodels import transform_numpy_array
from collections import deque
#import time

# Colours (BGR)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Check if ORB/SURF detector is available
try:
    _default_feature_extractor_ = image_feature_extractor.Orb().__class__
except AttributeError as attr_err:
    print attr_err
    try:
        _default_feature_extractor_ = image_feature_extractor.Surf().__class__
    except AttributeError as attr_err:
        print attr_err
        rospy.logfatal("No feature extractors available!")
        raise rospy.exceptions.ROSException(
            "Cannot initialise feature extractors")


def get_default_parameters():
    parameters = {}
    # CAMERA INFO
    # stereo_down position    : (0.6, 0.0, 0.4)
    # stereo_down orientation : (0.0, 0.0, 1.57)
    # stereo_front position   : (0.6, 0.0, 0.2)
    # stereo_front orientation: (1.57, 0.0, 1.57)
    # stereo_baseline: (0.12, 0.0, 0.0)
    parameters["visual_detector/camera_position"] = (0.6, 0.0, 0.2)
    parameters["visual_detector/camera_orientation"] = (1.57, 0.0, 1.57)
    parameters["visual_detector/camera_baseline"] = (0.12, 0.0, 0.0)
    parameters["visual_detector/camera_baseline_orientation"] = (0., 0., 0.)
    #
    # Simulator or real data
    parameters["visual_detector/panel/simulator"] = False
    # Template for simulator
    parameters["visual_detector/panel/simulator_template"] = (
        ["simulator_panel2.png"])
    parameters["visual_detector/panel/simulator_template_mask"] = []
    # Template for real data
    parameters["visual_detector/panel/real_template"] = (
        ["new_panel_template.png"])
    parameters["visual_detector/panel/real_template_mask"] = []
    # Use monocular/stereo detection
    parameters["visual_detector/panel/is_stereo"] = False
    # Camera topic
    parameters["visual_detector/panel/camera_root"] = "stereo_camera/left"
    # Image topic
    parameters["visual_detector/panel/image_topic"] = "image_rect_color"
    # Update rate
    parameters["visual_detector/panel/rate"] = -1
    #
    # PANEL GEOMETRY
    # Panel corners
    parameters["visual_detector/panel/corners"] = [
        [-0.4, 0.25, 0], [ 0.4,  0.25, 0], [0.4, -0.25, 0], [-0.4, -0.25, 0]]
    # Valve radius and centres
    parameters["visual_detector/valves/radius"] = 0.05
    parameters["visual_detector/valves/centres"] = [[-0.25, -0.125, 0.11],
        [ 0.0, -0.125, 0.11], [ 0.0, +0.125, 0.11], [+0.25, -0.125, 0.11]]
    return parameters

def set_default_parameters():
    parameters = get_default_parameters()
    for _param_name_ in parameters:
        rospy.set_param(_param_name_, parameters[_param_name_])



class VisualDetector(object):
    def __init__(self, name):
        self.name = name
        # Publish transforms
        self.tf_broadcaster = TransformBroadcaster()
        # ROS image message to cvimage convertor
        self.ros2cvimg = image_converter()
        
        #Create services
        self.enable_panel_valve_detection = metaclient.Service('/visual_detector/enable_panel_valve_detection', std_srvs.srv.Empty, self.enablePanelValveDetectionSrv, {})
        self.enable_valve_detection = metaclient.Service('/visual_detector/enable_valve_detection', std_srvs.srv.Empty, self.enableValveDetectionSrv, {})
        self.enable_chain_detection = metaclient.Service('/visual_detector/enable_chain_detection', std_srvs.srv.Empty, self.enableChainDetectionSrv, {})
        self.disable_panel_valve_detection = metaclient.Service('/visual_detector/disable_panel_valve_detection', std_srvs.srv.Empty, self.disablePanelValveDetectionSrv, {})
        self.disable_valve_detection = metaclient.Service('/visual_detector/disable_valve_detection', std_srvs.srv.Empty, self.disableValveDetectionSrv, {})
        self.disable_chain_detection = metaclient.Service('/visual_detector/disable_chain_detection', std_srvs.srv.Empty, self.disableChainDetectionSrv, {})
        
        
        camera_root = rospy.get_param("visual_detector/panel/camera_root")
        image_topic = rospy.get_param("visual_detector/panel/image_topic")
        is_stereo = rospy.get_param("visual_detector/panel/is_stereo")
        # Initialise image buffer
        self.image_buffer = camera_buffer(camera_root, image_topic, is_stereo)
        
        # Get camera details
        self._camera_ = STRUCT()
        cam_info_msg = self.image_buffer.get_camera_info(0)[0]
        self._camera_.frame_id = cam_info_msg.header.frame_id
        self._camera_.position = tuple(rospy.get_param(
            "visual_detector/camera_position"))
        self._camera_.orientation = quaternion_from_euler(
            *rospy.get_param("visual_detector/camera_orientation"))
        self._camera_.baseline = tuple(rospy.get_param(
            "visual_detector/camera_baseline"))
        self._camera_.baseline_orientation = quaternion_from_euler(
            *rospy.get_param("visual_detector/camera_baseline_orientation"))
        
        # Initialise the panel detector
        self.panel = STRUCT()
        self.init_panel_detector()
        # Publish panel position
        self.panel.pub = metaclient.Publisher('/visual_detector/valve_panel',
                                              Detection, {})
        self.panel.pose_msg_pub = metaclient.Publisher(
            '/pose_ekf_slam/landmark_update/panel_centre', 
            PoseWithCovarianceStamped, {})
        # Publish image of detected panel
        self.panel.img_pub = metaclient.Publisher(
            '/visual_detector/panel_img', Image, {})
        
        # Valve detection
        self.valve = STRUCT()
        self.valve.pub = STRUCT()
        self.valve.pub.img = metaclient.Publisher(
            "/visual_detector/valve_img", Image, {})
        
        # Each valve publisher is self.valve.pub.v{0..6}
        for valve_count in range(6):
            setattr(self.valve.pub, "v"+str(valve_count),
                    metaclient.Publisher(
                    '/visual_detector/valve'+str(valve_count), Detection, {}))
        
        # Create publisher
        self.pub_chain = metaclient.Publisher('/visual_detector/chain',
                                              Detection, {})
        
        rospy.timer.Timer(rospy.Duration(0.1), self.publish_transforms)
        # Enable panel detection by default
        self._enablePanelValveDetectionSrv_()
        print "Completed initialisation"

    def publish_transforms(self, *args, **kwargs):
        timestamp = rospy.Time.now()
        tf_br = self.tf_broadcaster
        camera = self._camera_
        frame_id = camera.frame_id
        
        tf_br.sendTransform(camera.position, camera.orientation, timestamp,
                            frame_id, 'girona500')
        tf_br.sendTransform(camera.baseline, camera.baseline_orientation,
                            timestamp, frame_id+'_right', frame_id)
    
    def init_panel_detector(self):
        panel = self.panel
        panel.init = False
        
        # Load parameters
        SIMULATOR = rospy.get_param("visual_detector/panel/simulator")
        IS_STEREO = rospy.get_param("visual_detector/panel/is_stereo")
        
        # Load template and mask according to SIMULATOR
        if SIMULATOR:
            template_file_list = rospy.get_param(
                "visual_detector/panel/simulator_template")
            template_mask_file_list = rospy.get_param(
                "visual_detector/panel/simulator_template_mask")
        else:
            template_file_list = rospy.get_param(
                "visual_detector/panel/real_template")
            template_mask_file_list = rospy.get_param(
                "visual_detector/panel/real_template_mask")
        
        # Load geometry
        panel_corners = np.asarray(rospy.get_param("visual_detector/panel/corners"))
        valve_radius = rospy.get_param("visual_detector/valves/radius")
        valve_centre = np.asarray(rospy.get_param("visual_detector/valves/centre"))
        
        # Get update rate
        update_rate = rospy.get_param("visual_detector/panel/rate")
        update_rate = update_rate if update_rate > 0 else None
        
        if IS_STEREO:
            panel.detector = objdetect.Stereo_detector(
                feat_detector=_default_feature_extractor_)
        else:
            panel.detector = objdetect.Detector(
                feat_detector=_default_feature_extractor_)
        # Set number of features from detector
        panel.detector.set_detector_num_features(4000)
        # Set flann ratio
        panel.detector.set_flann_ratio_threshold(0.6)
        
        # Get camera info msg and initialise camera
        cam_info = self.image_buffer.get_camera_info()
        panel.detector.init_camera(*cam_info)
        panel.detector.camera.set_tf_frame(self._camera_.frame_id,
                                           self._camera_.frame_id+"_right")
        # Set template
        panel.detector.set_corners3d(panel_corners)
        # Read template images
        for panel_template_file, panel_template_mask_file in (
            itertools.izip_longest(template_file_list, template_mask_file_list)):
            template_image_file = (roslib.packages.find_resource(
                "udg_pandora", panel_template_file))
            if len(template_image_file):
                template_image_file = template_image_file[0]
                template_image = cv2.imread(template_image_file, cv2.CV_8UC1)
            else:
                rospy.logerr("Could not locate panel template")
                raise rospy.exceptions.ROSException(
                        "Could not locate panel template")

            # Read template mask
            template_mask = None
            if not template_mask_file_list is None:
                template_mask_file = (roslib.packages.find_resource(
                    "udg_pandora", panel_template_mask_file))
            if len(template_mask_file):
                template_mask_file = template_mask_file[0]
                template_mask = cv2.imread(template_mask_file, cv2.CV_8UC1)
            
            panel.detector.add_to_template(template_image, template_mask)
        
        # Panel dimensions in pixels (from template)
        corners_2d = panel.detector._object_.corners_2d
        panel_width_px = corners_2d[1, 0] - corners_2d[0, 0]
        panel_height_px = corners_2d[2, 1] - corners_2d[1, 1]
        # Panel dimensions in m
        corners_3d = panel.detector._object_.corners_3d
        panel_width_m = np.abs(corners_3d[1, 0] - corners_3d[0, 0])
        panel_height_m = np.abs(corners_3d[2, 1] - corners_3d[1, 1])
        # Find how many pixels per m
        px_per_m = np.mean([panel_width_px/panel_width_m,
                            panel_height_px/panel_height_m])

        panel.valves = STRUCT()
        valves = panel.valves
        valves.px_per_m = px_per_m
        valves.radius = valve_radius
        valves.radius_px = valves.radius*px_per_m
        # Valve centres in homogenous coordinates
        valves.centre = np.asarray(valve_centre)
        valves.centre = np.hstack(
            (valves.centre, np.ones((valves.centre.shape[0], 1))))
        # Valve centre in the template in pixels
        valves.centre_px = (valves.centre[:, :2]*px_per_m +
                            [panel_width_px/2, panel_height_px/2])
        # Bounding box for each valve
        valves.bbox = np.array([[-valves.radius, -valves.radius, 0, 0],
                                [ valves.radius, -valves.radius, 0, 0],
                                [ valves.radius,  valves.radius, 0, 0],
                                [-valves.radius,  valves.radius, 0, 0]])
        valves.bbox_px = valves.bbox[:, :2]*px_per_m
        # Corner points of the valve handle
        valves.handle_corners = np.array([[-valves.radius, 0, 0, 0],
                                          [ valves.radius, 0, 0, 0]])

        # Buffer to store valve orientations
        valves.orientation = []
        for valve_count in range(valves.centre.shape[0]):
            valves.orientation.append(deque(maxlen=5))

        self.panel.callback_id = None
        self.panel.update_rate = update_rate
        
        # Create detection messages
        panel.detection_msg = Detection()
        panel.pose_msg = PoseWithCovarianceStamped()
        panel.pose_msg.header.frame_id = cam_info[0].header.frame_id
        

    def enablePanelValveDetectionSrv(self, req):
        self._enablePanelValveDetectionSrv_()
        print "Enabled panel detection"
        return std_srvs.srv.EmptyResponse()

    def _enablePanelValveDetectionSrv_(self):
        #self.panel.ts.registerCallback(self.detect_panel)
        self.panel.callback_id = (
            self.image_buffer.register_callback(self.detect_panel,
                                                self.panel.update_rate))
        #self.panel.sub = rospy.Subscriber("/stereo_camera/left/image_raw",
        #                                  sensor_msgs.msg.Image,
        #                                  self.detect_panel)

    def enableValveDetectionSrv(self, req):
        pass

    def enableChainDetectionSrv(self, req):
        pass

    def disablePanelValveDetectionSrv(self, req):
        #if not self.panel.sub is None:
        #    self.panel.sub.unregister()
        #self.panel.ts.callbacks = dict()
        self.image_buffer.unregister_callback(self.panel.callback_id)
        print "Disabled panel detection"
        return std_srvs.srv.EmptyResponse()

    def disableValveDetectionSrv(self, req):
        pass

    def disableChainDetectionSrv(self, req):
        pass

    def detect_panel(self, *args):
        #self.panel.sub.unregister()
        panel = self.panel
        if type(panel.detector) is objdetect.Detector:
            args = (args[0],)
        # Convert the images to BGR8 format
        _cvimage_ = [
            np.asarray(self.ros2cvimg.cvimage(_img_, COLOUR_FMT="bgr8")).copy()
            for _img_ in args]
        #try:
        #    cvimage = [self.panel.Retinex.retinex(_img_).astype(_img_.dtype)
        #        for _img_ in _cvimage_]
        #except:
        #    print "Error using retinex()"

        cvimage = _cvimage_
        panel.detector.detect(*cvimage)
        #self.panel.detector.show()
        panel_detected, panel_centre, panel_orientation = (
            panel.detector.location())
        panel_rpy = panel_orientation#[[2, 0, 1]]
        valid_panel_orientation = self._check_valid_panel_orientation_(panel_rpy)

        # Publish panel detection message
        time_now = rospy.Time.now()
        panel.detection_msg.header.stamp = time_now
        panel_detected = (panel_detected and valid_panel_orientation)
        panel_orientation_quaternion = quaternion_from_euler(*panel_rpy)
        panel.detection_msg.detected = panel_detected
        (panel.detection_msg.position.position.x,
         panel.detection_msg.position.position.y,
         panel.detection_msg.position.position.z) = panel_centre
        (panel.detection_msg.position.orientation.x,
         panel.detection_msg.position.orientation.y,
         panel.detection_msg.position.orientation.z,
         panel.detection_msg.position.orientation.w) = (
             panel_orientation_quaternion)
        panel.pub.publish(self.panel.detection_msg)

        if panel_detected:
            # Panel orientation is screwy - only broadcast the yaw
            # which is panel_orientation[1]
            panel.pose_msg.header.stamp = time_now
            (panel.pose_msg.pose.pose.position.x,
             panel.pose_msg.pose.pose.position.y,
             panel.pose_msg.pose.pose.position.z) = panel_centre
            (panel.pose_msg.pose.pose.orientation.x,
             panel.pose_msg.pose.pose.orientation.y,
             panel.pose_msg.pose.pose.orientation.z,
             panel.pose_msg.pose.pose.orientation.w) = (
                 panel_orientation_quaternion)
            cov = np.zeros((6, 6))
            pos_diag_idx = range(3)
            cov[pos_diag_idx, pos_diag_idx] = (
                (((1.2*np.linalg.norm(panel_centre))**2)*0.03)**2)
            panel.pose_msg.pose.covariance = cov.flatten().tolist()
            panel.pose_msg_pub.publish(panel.pose_msg)
            self.tf_broadcaster.sendTransform(tuple(panel_centre),
                panel_orientation_quaternion,
                time_now, "panel_centre", "stereo_front")

            # Detect valves if panel was detected at less than 2 metres
            self.detect_valves(panel.pose_msg)

            #print "\nPanel RPY = %s \n" % (panel_rpy*180/np.pi)
            print "Panel World Position: ", \
                np.squeeze(transform_numpy_array("world", "panel_centre",
                                                 np.zeros((1, 3))))
            print "Relative position   : ", panel_centre
            print "Standard deviation  : ", cov[range(3), range(3)]**0.5, "\n"

            bbox_colour = (255, 255, 255)
            corners = panel.detector.obj_corners.astype(np.int32)
            out_img = panel.detector.get_scene(0)
            cv2.polylines(out_img, [corners],
                          True, bbox_colour, 6)
        else:
            out_img = panel.detector.get_scene(0)

        # Publish image of detected panel
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(out_img), 
                                         encoding="bgr8")
        img_msg.header.stamp = time_now
        self.panel.img_pub.publish(img_msg)
        return
        #print "Detected = ", self.panel.detection_msg.detected
        #self._enablePanelValveDetectionSrv_()
        """
        Detect panel using contours
        # Filter the image by downscaling and upscaling
        src_img = cvimage[0].copy()
        filt_img = cv2.pyrDown(src_img, dstsize=(src_img.shape[1]/2, src_img.shape[0]/2))
        filt_img = cv2.pyrUp(filt_img, dstsize=(src_img.shape[1], src_img.shape[0]))
        #filt_img = src_img
        # Edge detection
        edges_img = cv2.Canny(filt_img, 0, 100, apertureSize=5, L2gradient=True)
        # Dilate the image
        dilate_img = cv2.dilate(edges_img, np.ones((5, 5)))
        # Extract contours
        contours, hierarchy = cv2.findContours(dilate_img, mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        def _angle_(pt1, pt2, pt0):
            dx1 = pt1[0] - pt0[0]
            dy1 = pt1[1] - pt0[1]
            dx2 = pt2[0] - pt0[0]
            dy2 = pt2[1] - pt0[1]
            return (dx1*dx2 + dy1*dy2)/np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10)
        detected_rects = []

        for idx in range(len(contours)):
            contour_length = cv2.arcLength(contours[idx], True)
            approx = cv2.approxPolyDP(contours[idx], contour_length*0.02, True)
            if ((np.squeeze(approx).shape[0] == 4) and
                    (np.fabs(cv2.contourArea(approx)) > 5000) and
                    cv2.isContourConvex(approx)):
                maxCosine = 0
                for j in range(2, 5):
                    cosine = np.fabs(_angle_(approx[j%4][0], approx[j-2][0], approx[j-1][0]))
                    maxCosine = max((maxCosine, cosine))
                if( maxCosine < 0.3 ):
                    detected_rects.append(approx)

        out_img = self.panel.detector.get_scene(0).copy()
        color = (255, 255, 255)
        cv2.drawContours(out_img, detected_rects, -1, color, 3)
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(edges_img))
        img_msg.header.stamp = time_now
        self.panel.img_pub.publish(img_msg)
        #code.interact(local=locals())
        return
        """

    def _check_valid_panel_orientation_(self, panel_rpy):
        # if panel_orientation{0,2} not close to zero, return false
        abs_rpy = np.abs(panel_rpy)
        if (((abs_rpy[0] < 0.35) or ((np.pi-abs_rpy[0]) < 0.35))):
            #and ((abs_rpy[1] < 0.35) or ((np.pi-abs_rpy[1]) < 0.35))):
            return True
        else:
            return False
        #remainder = np.mod(np.pi - np.mod(abs_po, np.pi), np.pi)
        #if np.any(remainder > 0.2):
        #    return False
        #else:
        #    return True
        #return True

    def detect_valves(self, detection_msg):
        if type(detection_msg) is Detection:
            panel_centre = np.array((detection_msg.position.position.x,
                                     detection_msg.position.position.y,
                                     detection_msg.position.position.z))
            panel_orientation_quaternion = np.asarray(
                (detection_msg.position.orientation.x,
                 detection_msg.position.orientation.y,
                 detection_msg.position.orientation.z,
                 detection_msg.position.orientation.w))
            panel_detected = detection_msg.detected
        elif type(detection_msg) is PoseWithCovarianceStamped:
            panel_detected = True
            panel_centre = np.asarray(
                (detection_msg.pose.pose.position.x,
                 detection_msg.pose.pose.position.y,
                 detection_msg.pose.pose.position.z))
            panel_orientation_quaternion = np.asarray(
                (detection_msg.pose.pose.orientation.x,
                 detection_msg.pose.pose.orientation.y,
                 detection_msg.pose.pose.orientation.z,
                 detection_msg.pose.pose.orientation.w))
            panel_orientation_euler = np.asarray(
                euler_from_quaternion(panel_orientation_quaternion))
        # Get location of valves if closer than ~2m and orientation<30 degrees
        # Panel positions (in m) are (relative to 0,0 centre):
        #   left: (-0.5, +-0.25), centre: (0, +-0.25), right: (0.5, +-0.25)
        # Stem length is 10-11cm, t-bar length is 10cm
        scene = self.panel.detector.get_scene(0)
        # Equalise the histogram
        #scene = cv2.equalizeHist(scene)
        # Get transformation of panel centre from
        # (x=0, y=0, z=0, r=0, p=0, y=0), (panel_centre, panel_orientation)
        detected_valve_orientations = None
        if (panel_detected and (np.linalg.norm(panel_centre) <= 2) and
            (np.abs(panel_orientation_euler[1]) < 0.5)):
            # Wait until transform becomes available
            homogenous_rotation_matrix = (
                quaternion_matrix(panel_orientation_quaternion))
            homogenous_rotation_matrix[:3, -1] = panel_centre
            """
            try:
                self.valve.tflistener.waitForTransform("bumblebee2",
                    "/panel_centre", rospy.Time(), rospy.Duration(0.5))
            except tf.Exception:
                rospy.logerr("DETECT_VALVES(): Could not get transform from \
                panel_centre to bumblebee2")
                return
            """
            time_now = detection_msg.header.stamp #rospy.Time.now()
            valves = self.panel.valves
            valve_msg = Detection()
            valve_msg.header.stamp = time_now
            valve_msg.header.frame_id = "panel_centre"
            # List containing each valve centre and orientation
            detected_valve_orientations = []
            for valve_idx in range(valves.centre.shape[0]):
                # Get bounding box for the current valve
                this_valve = valves.centre[valve_idx] + valves.bbox
                # Convert points from panel to camera co-ordinates
                this_valve_camcoords = np.dot(homogenous_rotation_matrix,
                                              this_valve.T).T
                # Project the 3D points from camera coordinates to pixels
                px_corners = self.panel.detector.camera.project3dToPixel(
                    this_valve_camcoords[:, :3])
                px_corners = px_corners.astype(np.int32)
                left, top = np.min(px_corners, axis=0)
                right, bottom = np.max(px_corners, axis=0)
                # Only proceed if the bounding box is within the image
                if (left < 0 or top < 0 or 
                    right > scene.shape[1] or bottom > scene.shape[0]):
                    pass
                else:
                    # Get approximate length of the valve in pixels
                    bbox_side_lengths = ((np.diff(np.vstack((px_corners, px_corners[0])), axis=0)**2).sum(axis=1))**0.5
                    valve_length_px = np.int32(0.8*np.mean(bbox_side_lengths))
                    valve_im_bbox = (left, top, right, bottom)
                    this_valve_orientation = self.detect_valve_orientation(
                        scene, valve_im_bbox, HoughMinLineLength=valve_length_px,
                        HoughThreshold=int(0.9*valve_length_px),
                        HIGHLIGHT_VALVE=True)

                    if not this_valve_orientation is None:
                        valve_msg.detected = True
                        (valve_msg.position.position.x,
                         valve_msg.position.position.y,
                         valve_msg.position.position.z) = valves.centre[valve_idx][:3]
                        # Obtain rotation of the valve wrt the panel
                        # TODO: Find a more robust way of doing this
                        panel_rpy = panel_orientation_euler
                        valve_rpy = (0, 0, this_valve_orientation - panel_rpy[2])
                        valve_orientation_quaternion = quaternion_from_euler(*valve_rpy)
                        (valve_msg.position.orientation.x,
                         valve_msg.position.orientation.y,
                         valve_msg.position.orientation.z,
                         valve_msg.position.orientation.w) = valve_orientation_quaternion
                        v_pub = getattr(self.valve.pub, "v"+str(valve_idx))
                        v_pub.publish(valve_msg)
                        self.tf_broadcaster.sendTransform(tuple(valves.centre[valve_idx]),
                            valve_orientation_quaternion, time_now,
                            "valve"+str(valve_idx), "panel_centre")
                        valves.orientation[valve_idx].appendleft(this_valve_orientation)
                    else:
                        # Did not detect valve orientation
                        # Overlay the previous estimation
                        this_valve_handle_orientation = np.median(
                            valves.orientation[valve_idx])
                        if not np.isnan(this_valve_handle_orientation):
                            # Check if previous estimate available
                            this_valve_handle_orientation = (
                                valves.orientation[valve_idx][0])
                            # Get corners of handle with zero rotation and
                            # rotate by this_valve_orientation
                            valve_rotation_matrix = euler_matrix(
                                0, 0, -this_valve_handle_orientation)
                            this_valve_handle = np.dot(valve_rotation_matrix, valves.handle_corners.T).T
                            # Offset the valve by the valve centre
                            this_valve_handle += valves.centre[valve_idx]
                            # Get handle coordinates in camera frame
                            this_handle_camcoords = np.dot(homogenous_rotation_matrix, this_valve_handle.T).T

                            # Project the 3D points from camera coordinates to pixels
                            handle_px_corners = self.panel.detector.camera.project3dToPixel(this_handle_camcoords[:, :3])
                            handle_px_corners = handle_px_corners.astype(np.int32)
                            cv2.line(scene, tuple(handle_px_corners[0].astype(int)),
                                     tuple(handle_px_corners[1].astype(int)),
                                     RED, 5)

                        #detected_valve_orientations.append(None)
                        #valve_centre_3d = transform_numpy_array("bumblebee2", "panel_centre", valve_centre[np.newaxis, valve_idx])
                        #v_centre_px = self.panel.detector.camera.project3dToPixel(valve_centre_3d)[0]
                        #try:
                        #    scene[v_centre_px[1], v_centre_px[0]] = 255
                        #    scene[px_corners[:, 1], px_corners[:, 0]] = 255
                        #except IndexError:
                        #    pass

                #cv2.polylines(scene, [px_corners], True, (255, 255, 255), 2)
        img_msg = self.ros2cvimg.img_msg(cv2.cv.fromarray(scene), encoding="bgr8")
        self.valve.pub.img.publish(img_msg)
        return detected_valve_orientations

    def detect_valve_orientation(self, img, bbox=None,
        HIGHLIGHT_VALVE=False, CannyThreshold1=50, CannyThreshold2=100,
        HoughRho=1, HoughTheta=np.pi/180, HoughThreshold=40,
        HoughMinLineLength=60, HoughMaxLineGap=5):
        """detect_valve_orientation(self, img, bbox=None) -> theta
        Estimate valve orientation using Hough transform:
        Estimate the orientation of the valve in the image within the specified
        bounding box (left, top, right, bottom) in pixels. If bbox==None, the
        entire image is used.
        """
        im_box = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        im_edges = cv2.Canny(im_box, CannyThreshold1, CannyThreshold2)
        # Zero lines in the borders using a circular mask
        box_height, box_width = im_box.shape[:2]
        mask = np.zeros(im_box.shape[:2])
        mask_centre = (int(box_width/2), int(box_height/2))
        mask_radius = int(max((box_height, box_width))/2*1.1)
        cv2.circle(mask, mask_centre, mask_radius, WHITE, thickness=-1)
        mask[mask>0] = 1
        # mask the 1st and 3rd quadrants since these angles are not possible
        centre_border_px = 0.15*HoughMinLineLength
        cv2.rectangle(mask, (0, box_height),
            (int((box_width/2)-centre_border_px), int((box_height/2)+centre_border_px)),
            (0, 0, 0), thickness=-1)
        cv2.rectangle(mask, (box_width, 0),
            (int((box_width/2)+centre_border_px), int((box_height/2)-centre_border_px)),
            (0, 0, 0), thickness=-1)
        im_edges *= mask
        lines = np.squeeze(cv2.HoughLinesP(im_edges, HoughRho, HoughTheta,
            HoughThreshold, minLineLength=HoughMinLineLength,
            maxLineGap=HoughMaxLineGap))
        # if multiple lines, choose the longest
        if lines.ndim == 1:
            valve_points = lines
        elif lines.ndim == 2:
            pts1 = lines[:, :2]
            pts2 = lines[:, 2:]
            line_lengths = np.sqrt(np.power(pts2 - pts1, 2).sum(axis=1))
            valve_idx = line_lengths.argmax()
            valve_points = lines[valve_idx]
        else:
            return None #raise ValueError
        v_pt1 = valve_points[:2]
        v_pt2 = valve_points[2:]
        delta_xy = v_pt2 - v_pt1
        if HIGHLIGHT_VALVE and valve_points.shape[0]:
            if not bbox is None:
                v_pt1 += [bbox[0], bbox[1]]
                v_pt2 += [bbox[0], bbox[1]]
                cv2.line(img, tuple(v_pt1), tuple(v_pt2), BLUE, 6)

        # Get the angle using arctan
        valve_orientation = np.arctan2(delta_xy[1], delta_xy[0])
        if -92.5 < valve_orientation*180/np.pi < -87.5:
            valve_orientation = -valve_orientation
        if -2.5 < valve_orientation < 92.5:
            return valve_orientation
        else:
            return None
    

if __name__ == '__main__':
    try:
        import subprocess
        # Load ROS parameters
        config_file_list = roslib.packages.find_resource("udg_pandora", "visual_detector.yaml")
        if len(config_file_list):
            config_file = config_file_list[0]
            subprocess.call(["rosparam", "load", config_file])
        else:
            print "Could not locate visual_detector.yaml, using default parameters"
            set_default_parameters()
        
        rospy.init_node('visual_detector')
        visual_detector = VisualDetector(rospy.get_name())
        #remap_camera_name = rospy.resolve_name(rospy.get_param("/visual_detector/camera_root"))
        #rospy.loginfo("Using camera root: " + remap_camera_name)
        #remap_image_name = rospy.resolve_name(rospy.get_param("/visual_detector/image_topic"))
        #rospy.loginfo("Using image topic: " + remap_image_name)
        #visual_detector = VisualDetector(rospy.get_name(),
        #                                 remap_camera_name, remap_image_name)
        rospy.spin()
    except rospy.ROSInterruptException: pass
