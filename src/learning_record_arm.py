#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
import std_msgs.msg
import std_srvs.srv

#use to load the configuration function
import cola2_ros_lib

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped

import numpy as np

#import to use mutex
import threading

class LearningRecordArm:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = PoseStamped()
        self.armPose = PoseStamped()
        self.goalInit = False

        self.lock = threading.Lock()
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        rospy.Subscriber("/arm/valve_pose", PoseStamped, self.updateGoalPose)

    def getConfig(self):
        param_dict = {'filename': 'learning/record/filename',
                      'numberSample': 'learning/record/number_sample'
                      }
        cola2_ros_lib.getRosParams(self, param_dict)
        self.file = open( self.filename + "_" + str(self.numberSample) +".csv", 'w')

    def updateArmPose(self, armPose):
        if self.goalInit :
            self.lock.acquire()
            try:
                self.armPose = armPose
                #Compute the distance
                #rospy.loginfo('X Diference ' + str(self.armPose.pose.position.x - self.goalPose.pose.position.x ) )
                s = (repr(self.goalPose.pose.position.x - self.armPose.pose.position.x )
                +" "+ repr(self.goalPose.pose.position.y - self.armPose.pose.position.y )
                +" "+ repr(self.goalPose.pose.position.z - self.armPose.pose.position.z )
                +"\n")
                self.file.write(s)
            finally:
                self.lock.release()
        else :
            rospy.loginfo('Waiting to initialize the valve position')


    def updateGoalPose(self, goalPose):
        self.lock.acquire()
        try:
            self.goalPose = goalPose
            self.goalInit = True
        finally:
            self.lock.release()


if __name__ == '__main__':
    try:
        rospy.init_node('learning_record_arm')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record_arm = LearningRecordArm( rospy.get_name() )
        rospy.spin()
    except rospy.ROSInterruptException: pass