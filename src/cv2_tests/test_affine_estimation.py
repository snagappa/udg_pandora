#!/usr/bin/env python
import cv2
import numpy as np

def computeAffine3d():
    src_points = cv2.cv.CreateMat(1, 4, cv2.cv.CV_32FC3)
    dst_points = cv2.cv.CreateMat(1, 4, cv2.cv.CV_32FC3)
    cv2.cv.Zero(src_points)
    cv2.cv.Zero(dst_points)
    src_points = np.array( src_points )
    dst_points = np.array( dst_points )
    	
    # Set src_points to (0,0,0), (1,0,0), (0,1,0) and (0,0,1)
    # Set dst_points to (1,0,0), (2,0,0), (1,1,0) and (1,0,1) --> x + 1
    src_points[0][1][0] = 1.0
    src_points[0][2][1] = 1.0
    src_points[0][3][2] = 1.0 
    dst_points[0][0][0] = 1.0
    dst_points[0][1][0] = 2.0
    dst_points[0][2][0] = 1.0
    dst_points[0][2][1] = 1.0
    dst_points[0][3][0] = 1.0
    dst_points[0][3][2] = 1.0
    
    ok_flag, H, status1 = cv2.estimateAffine3D(src_points, dst_points)
    
    print ok_flag
    print H
    print status1
    

if __name__ == '__main__':
	computeAffine3d()
