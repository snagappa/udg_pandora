#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy

from numpy import *
from udg_pandora.msg import Detection

class GeometricDetector:
    def __init__(self, name):
        self.name = name
        # Default parameters
        self.f = 80 # mm
        self.pixels_mm_x = 6.96 # px/mm
        self.pixels_mm_y = 8.9 # px/mm
        self.image_x_size = 640 # px
        self.image_y_size = 480 # px
        self.camera_position = [0, 0, -1] # m
        self.camera_orientation = [0, pi/2, 0] # rad
        self.visibility = 2.0
        
        self.number_points = 3 # panle, valve, chain
        self.wPw = matrix(zeros(4*self.number_points)).reshape(4, self.number_points)
        
        panel = Detection()
        valve = Detection()
        chain = Detection()
        panel.detected = False
        valve.detected = False
        chain.detected = False
        self.detected_objects = [panel, valve, chain]
         
        self.getConfig()
        
        # Compute camera intrinsic matrix
        self.intrinsics = matrix([self.f*self.pixels_mm_x, 0, self.image_x_size/2, 0, 0, self.f*self.pixels_mm_y, self.image_y_size/2, 0, 0, 0, 1, 0]).reshape(3,4)
        
        # Camera transformation wrt vehicle
        vTc = matrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, self.camera_position[0], self.camera_position[1], self.camera_position[2], 1]).reshape(4,4)
        vRc = self.rotationRPY(self.camera_orientation[0], self.camera_orientation[1], self.camera_orientation[2])
        self.v_transf_c = vTc.T * vRc
        
        
    def getConfig(self) :
        """ Load parameters from the rosparam server """
        
        if rospy.has_param('visual_detector/f'):
            self.f = rospy.get_param('visual_detector/f')
        else:
            rospy.logfatal('visual_detector/f')
        
        if rospy.has_param('visual_detector/pixels_mm_x'):
            self.pixels_mm_x = rospy.get_param('visual_detector/pixels_mm_x')
        else:
            rospy.logfatal('visual_detector/pixels_mm_x')
        
        if rospy.has_param('visual_detector/pixels_mm_y'):
            self.pixels_mm_y = rospy.get_param('visual_detector/pixels_mm_y')
        else:
            rospy.logfatal('visual_detector/pixels_mm_y')
        
        if rospy.has_param('visual_detector/image_x_size'):
            self.image_x_size = rospy.get_param('visual_detector/image_x_size')
        else:
            rospy.logfatal('visual_detector/image_x_size')
            
        if rospy.has_param('visual_detector/image_y_size'):
            self.image_y_size = rospy.get_param('visual_detector/image_y_size')
        else:
            rospy.logfatal('visual_detector/image_y_size')
        
        if rospy.has_param('visual_detector/camera_position'):
            self.camera_position = rospy.get_param('visual_detector/camera_position')
        else:
            rospy.logfatal('visual_detector/camera_position')
        
        if rospy.has_param('visual_detector/camera_orientation'):
            self.camera_orientation = rospy.get_param('visual_detector/camera_orientation')
        else:
            rospy.logfatal('visual_detector/camera_orientation')
 
        if rospy.has_param('visual_detector/visibility'):
            self.visibility = rospy.get_param('visual_detector/visibility')
        else:
            rospy.logfatal('visual_detector/visibility')
        
        if rospy.has_param('visual_detector/panel_position'):
            panel_position = rospy.get_param('visual_detector/panel_position')
            self.wPw[0,0] = panel_position[0]
            self.wPw[1,0] = panel_position[1]
            self.wPw[2,0] = panel_position[2]
            self.wPw[3,0] = 1
        else:
            rospy.logfatal('visual_detector/panel_position')
            
        if rospy.has_param('visual_detector/valve_position'):
            valve_position = rospy.get_param('visual_detector/valve_position')
            self.wPw[0,1] = valve_position[0]
            self.wPw[1,1] = valve_position[1]
            self.wPw[2,1] = valve_position[2]
            self.wPw[3,1] = 1
        else:
            rospy.logfatal('visual_detector/valve_position')
            
        if rospy.has_param('visual_detector/chain_position'):
            chain_position = rospy.get_param('visual_detector/chain_position')
            self.wPw[0,2] = chain_position[0]
            self.wPw[1,2] = chain_position[1]
            self.wPw[2,2] = chain_position[2]
            self.wPw[3,2] = 1
        else:
            rospy.logfatal('visual_detector/chain_position')
        
        
    def computeCameraExtrinsics(self, vehicle_pose, vehicle_orientation):
        wTv = matrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, vehicle_pose[0], vehicle_pose[1], vehicle_pose[2], 1]).reshape(4,4)
        wRv = self.rotationRPY(vehicle_orientation[0], vehicle_orientation[1], vehicle_orientation[2]) 
        w_transf_v = wTv.T * wRv
        w_transf_c = w_transf_v * self.v_transf_c;
        
        # compute inverse of w_transf_c  --> c_transf_w
        r = w_transf_c[0:3,0:3].T
        t = w_transf_c[0:3,3]
        temp = concatenate((r,-r*t), axis=1)
        a = matrix([0, 0, 0, 1]).reshape(1,4)
        c_transf_w = concatenate((temp, a), axis=0)
        
        return c_transf_w
    
        
    def detectObjects(self, vehicle_pose, vehicle_orientation):
        extrinsics = self.computeCameraExtrinsics(vehicle_pose, vehicle_orientation)
        
        cPu = matrix(zeros(2*self.number_points)).reshape(2, self.number_points)
        for i in range(self.number_points): 
            ret = self.intrinsics * extrinsics * self.wPw[:,i]
            cPu[0,i] = ret[0]/ret[2]
            cPu[1,i] = ret[1]/ret[2]
         
        # Check deyecyed points
	print 'Check for objects...'
        for i in range(self.number_points): 
            if cPu[0,i] > 0 and cPu[0,i] < self.image_x_size and cPu[1,i] > 0 and cPu[1,i] < self.image_y_size:
                object_wrt_camera = extrinsics*self.wPw[:,i]

		# Minimum focus distance
                if object_wrt_camera[2,0] > 0.25: 
              	    print 'detected object: ', i
                    object_wrt_camera = extrinsics*self.wPw[:,i]
                    distance = sqrt(object_wrt_camera[0]**2 + object_wrt_camera[1]**2 + object_wrt_camera[2]**2)
                    if distance > self.visibility:
                        print 'too far'
                        self.detected_objects[i].detected = False
                    else:
                        self.detected_objects[i].detected = True
                        self.detected_objects[i].position.position.x = object_wrt_camera[0,0]
                        self.detected_objects[i].position.position.y = object_wrt_camera[1,0]
                        self.detected_objects[i].position.position.z = object_wrt_camera[2,0]
                else:
                    print 'the object is just behind the camera or too close to it'    
            else:
                self.detected_objects[i].detected = False
        
        return self.detected_objects


    def rotationRPY(self, r, p, y):
        ret = matrix([ cos(p)*cos(y),  -cos(r)*sin(y)+sin(r)*sin(p)*cos(y),    sin(r)*sin(y)+cos(r)*sin(p)*cos(y),     0,
        cos(p)*sin(y),  cos(r)*cos(y)+sin(r)*sin(p)*sin(y),     -sin(r)*cos(y)+cos(r)*sin(p)*sin(y),    0,
        -sin(p),        sin(r)*cos(p),                          cos(r)*cos(p),                          0,
        0,              0,                                      0,                                      1]).reshape(4,4)
        
        return ret

    
if __name__ == '__main__':
    visual_detector = GeometricDetector('geometric_detector')
    v_pos = [5, 0, 0];
    v_or = [0, 0, pi/2];
    d = visual_detector.detectTargets(v_pos, v_or)
    print 'detected points: ', d
