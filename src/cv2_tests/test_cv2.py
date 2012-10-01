#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class TestCV2:
    def __init__(self, name):
        self.name = name
        
        self.image_pub = rospy.Publisher("image_modified", Image)
        cv2.cv.NamedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/stereo_down/left/image_rect_color", Image, self.callback)
        
        self.orbDetector = cv2.FeatureDetector_create("ORB")
        self.orbDescriptorExtractor = cv2.DescriptorExtractor_create("ORB")
        self.matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming') # 'BruteForce-Hamming' # FlannBased
        
        self.old_key = None
        self.old_desc = None
        self.old_img = None
        
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv(data, "bgr8")
            img_array = np.asarray(cv_image)
        except CvBridgeError, e:
            print e

        img_array, key, desc = self.featuresAndDescriptors(img_array)
        
        if desc != None and self.old_desc != None:
            ret = self.match(self.old_key, key, self.old_desc, desc)
            if ret != None:
                # More than 10 features have been detected in both images
                img_array, H = self.computeHomography(ret[0], ret[1], img_array)
                
        cv2.imshow("Image window", img_array)
        cv2.cv.WaitKey(3)
        
        self.old_key = key
        self.old_desc = desc
        self.old_img = img_array
        
        try:
            self.image_pub.publish(self.bridge.cv_to_imgmsg(cv_image, "rgb8"))
        except CvBridgeError, e:
            print e


    def featuresAndDescriptors(self, img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = self.orbDetector.detect(img_grey)

        # Print features
#        for p in keypoints:
#            x, y = np.int32(p.pt)
#            r = int(0.5 * p.size)
#            cv2.circle(img, (x, y), r, (0, 255, 0))

        (keypoints, desc) = self.orbDescriptorExtractor.compute(img_grey, keypoints)
#        if desc != None:
#            for i in desc:
#                print i
        return img, keypoints, desc


    def match(self, old_key, key, old_desc, desc):
        raw_matches = self.matcher.knnMatch(desc, old_desc, 2)
        threshold = 0.7
        eps = 1e-5
        matches = [(m1.trainIdx, m1.queryIdx) for m1, m2 in raw_matches if (m1.distance+eps) / (m2.distance+eps) < threshold]
        # print matches 
        
        if len(matches) > 10:
            p0 = np.float32( [old_key[i].pt for i, j in matches] )
            p1 = np.float32( [key[j].pt for i, j in matches] )
            return [p0, p1]
        
        return None
        
        
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
        test_cv2 = TestCV2(rospy.get_name())
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down"
            cv2.cv.DestroyAllWindows()
    except rospy.ROSInterruptException: pass
