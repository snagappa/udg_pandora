#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class PointFeature:
    def __init__(self, pose_image, descriptor):
        self.pose_image = pose_image
        self.descriptor = descriptor
        self.stereo_match_index = None
        self.epipolar_filter = False
        self.last_image_match_index = None
        self.pose_3d = None
        
        
class StereoTestCV2:
    def __init__(self, name):
        self.name = name
        
        self.image_pub = rospy.Publisher("image_modified", Image)
        cv2.cv.NamedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_sub_left = rospy.Subscriber("/stereo_down/left/image_rect_color", Image, self.callbackLeft)
        self.image_sub_right = rospy.Subscriber("/stereo_down/right/image_rect_color", Image, self.callbackRight)
        
        self.orbDetector = cv2.FeatureDetector_create("ORB")
        self.orbDescriptorExtractor = cv2.DescriptorExtractor_create("ORB")
        self.matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming') # 'BruteForce-Hamming' # FlannBased
        
        self.left_key = None
        self.left_desc = None
        self.left_img = None
        self.right_key = None
        self.right_desc = None
        self.right_img = None
        self.last_right_key = None
        self.last_right_desc = None
        self.last_right_img = None
        self.last_right_3d_points = None
        
        
    def callbackLeft(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv(data, "bgr8")
            self.left_img = cv2.pyrDown(np.asarray(cv_image))
        except CvBridgeError, e:
            print e

        self.left_img, self.left_key, self.left_desc = self.featuresAndDescriptors(self.left_img)
        

    def callbackRight(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv(data, "bgr8")
            self.right_img = cv2.pyrDown(np.asarray(cv_image))
        except CvBridgeError, e:
            print e

        self.right_img, self.right_key, self.right_desc = self.featuresAndDescriptors(self.right_img)
        
        features = []
        for i, j in zip(self.right_key, self.right_desc):
            f = PointFeature(i.pt, j)
            features.append(f)
        
        
        match_stereo = self.match(self.left_key, self.right_key, self.left_desc, self.right_desc, features, 'stereo')
        print 'pose_image: ', features[10].pose_image
        print 'desc: ', features[10].descriptor
        print 'stereo_match: ', features[10].stereo_match_index
        print 'epipolar_filter: ', features[10].epipolar_filter
        
        
        if match_stereo != None:
            points_3d = self.obtain3DPoints(match_stereo[0].T, match_stereo[1].T)
        
        # If 3D points in current and last images
        if points_3d != None and self.last_right_3d_points != None:
            match_previous = self.match(self.last_right_key, self.right_key, self.last_right_desc, self.right_desc, features, 'last_image')
            if match_previous != None:
                pass
            
        # Save current keys, features and 3D points
        self.last_right_key = self.right_key
        self.last_right_desc = self.right_desc
        self.last_right_img = self.right_img
        self.last_right_3d_points = points_3d
        
    def featuresAndDescriptors(self, img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = self.orbDetector.detect(img_grey)
        (keypoints, desc) = self.orbDescriptorExtractor.compute(img_grey, keypoints)
        return img, keypoints, desc
        
        
    def epipolarFilter(self, p0, p1):
        p0_temp = p0.tolist()
        p1_temp = p1.tolist()
        
        filter_p0 = [i for i, j in zip(p0_temp, p1_temp) if (i[1] - j[1]) < 1.5]
        filter_p1 = [j for i, j in zip(p0_temp, p1_temp) if (i[1] - j[1]) < 1.5]
      
         # Print features matched in both images
        for i, j in zip(filter_p0, filter_p1):
            cv2.line(self.right_img, (int(i[0]), int(i[1])), (int(j[0]), int(j[1])), (255,255,255))
            
        cv2.imshow("Image window", self.right_img)
        cv2.cv.WaitKey(3)

        return np.array(filter_p0), np.array(filter_p1)
        
        
    def computeDisparity(self):
        window_size = 6
        min_disp = 16
        num_disp = 112-min_disp
        stereo = cv2.StereoSGBM(minDisparity = min_disp, 
                                numDisparities = num_disp, 
                                SADWindowSize = window_size,
                                uniquenessRatio = 10,
                                speckleWindowSize = 100,
                                speckleRange = 32,
                                disp12MaxDiff = 1,
                                P1 = 8*3*window_size**2,
                                P2 = 32*3*window_size**2,
                                fullDP = False
                                )

        print 'computing disparity...'
        disp = stereo.compute(self.left_img, self.right_img).astype(np.float32) / 16.0
        cv2.imshow('disparity', (disp-min_disp)/num_disp)
        cv2.waitKey(3)
        
        
    def match(self, old_key, key, old_desc, desc, features, match_mode):
        raw_matches = self.matcher.knnMatch(desc, old_desc, 2)
        threshold = 0.7
        eps = 1e-5
        matches = [(m1.trainIdx, m1.queryIdx) for m1, m2 in raw_matches if (m1.distance+eps) / (m2.distance+eps) < threshold]
        
        if match_mode == 'stereo':
            for i, j in matches:
                features[j].stereo_match_index = i
                if abs(old_key[i].pt[1] - key[j].pt[1]) < 1.5:
                    features[j].epipolar_filter = True
        elif match_mode == 'last_image':
            for i, j in matches:
                features[j].last_image_match_index = i
        else:
            print 'invalid match mode'
            
        if len(matches) > 10:
            p0 = np.float32( [old_key[i].pt for i, j in matches] )
            p1 = np.float32( [key[j].pt for i, j in matches] )
            return [p0, p1, matches]
        
        return None
        
        
    def obtain3DPoints(self, p0, p1):
        proj_left = np.matrix([724.2 ,    0.  ,  520.08,    0., 0.  ,  722.33,  385.77,    0.  , 0.  ,    0.  ,    1.  ,    0.]).reshape(3,4)
        proj_right = np.matrix([724.2 ,    0.  ,  520.08,    -82.25, 0.  ,  722.33,  385.77,    0.  , 0.  ,    0.  ,    1.  ,    0.]).reshape(3,4)
        points_3d = cv2.triangulatePoints(proj_left, proj_right, p0, p1)
        
        # TODO: Check--> cv2.convertPointsFromHomogeneous(points_3d)
        ret = []
        for i in points_3d.T:
            a = i / i[3]
            ret.append(a)
        return ret
    
        
    def computeHomography(self, p0, p1, img):
        green, red = (0, 255, 0), (0, 0, 255)
        inliner_n = 0
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 10.0)
        inlier_n = sum(status)
        if inlier_n > 10:
            # print H
            # Draw a green line between detected features
            for (x1, y1), (x2, y2), inlier in zip(np.int32(p0), np.int32(p1), status):
                    cv2.line(img, (x1, y1), (x2, y2), (red, green)[inlier])
                    
            # h, w = img.shape[:2]
            # overlay = cv2.warpPerspective(self.old_img, H, (w, h))
            # img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0)
 
        return [img, H]


if __name__ == '__main__':
    try:
        rospy.init_node('test_cv2')
        stereo_test_cv2 = StereoTestCV2(rospy.get_name())
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down"
            cv2.cv.DestroyAllWindows()
    except rospy.ROSInterruptException: pass
