#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
import std_msgs.msg
import std_srvs.srv

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped
#include message of the ekf giving the valve position
from geometry_msgs.msg import PoseWithCovarianceStamped
#include message of the ekf giving the position of the robot
from nav_msgs.msg import Odometry

#include message of the pose_ekf_slam.
from pose_ekf_slam.msg import Map
#include message for the pose of the landmark
from geometry_msgs.msg import Point

from tf.transformations import euler_from_quaternion

#import to use mutex
import threading

class LearningRecord:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        self.goalPose = Point()
        self.robotPose = Odometry()
        self.lock = threading.Lock()
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)
        #rospy.Subscriber("/pose_ekf_slam/landmark_update/valve_1", PoseWithCovarianceStamped, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.updateGoalPose)
        rospy.Subscriber("/pose_ekf_slam/odometry", Odometry, self.updateRobotPose )


    def getConfig(self):
       if rospy.has_param('learning/record/filename') :
           self.filename = rospy.get_param('learning/record/filename')
       else :
           rospy.logerr('Prameter filename not found')

       if rospy.has_param('learning/record/number_sample') :
           self.numberSample = rospy.get_param('learning/record/number_sample')
           print 'number samples '+ str(self.numberSample)
       else :
           rospy.logerr('Prameter nbDataRepro not found')

       if rospy.has_param('learning/record/landmark_id') :
           self.landmark_id = rospy.get_param('learning/record/landmark_id')
       else :
           rospy.logerr('Prameter landmark_id not found')


       self.file = open( self.filename + "_" + str(self.numberSample) +".csv", 'w')

    def updateArmPose(self, armPose):
#        euler = euler_from_quaternion( armPose.pose.orientation, 'sxyz' )  #,  axes='sxyz' );
        quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y, armPose.pose.orientation.z, armPose.pose.orientation.w ]
        euler = euler_from_quaternion( quaternion, 'sxyz' )
        #s = repr(armPose.pose.position.x)+" "+ repr(armPose.pose.position.y) + " " + repr(armPose.pose.position.x) +" "+ repr(euler[0])  +" "+ repr(euler[1])  +" "+ repr(euler[2]) +"\n"

        self.lock.acquire()
        try:
#            s = repr(self.goalPose.x - (armPose.pose.position.x + self.robotPose.pose.pose.position.x  ) )+" "+ repr( self.goalPose.y - (armPose.pose.position.y + self.robotPose.pose.pose.position.y ) ) + " " + repr( self.goalPose.z - ( armPose.pose.position.z + self.robotPose.pose.pose.position.z ) ) +" "+ repr(euler[0])  +" "+ repr(euler[1])  +" "+ repr(euler[2]) +"\n"
            s = repr((armPose.pose.position.x + self.robotPose.pose.pose.position.x ) - self.goalPose.x )+" "+ repr( (armPose.pose.position.y + self.robotPose.pose.pose.position.y ) - self.goalPose.y ) + " " + repr( ( armPose.pose.position.z + self.robotPose.pose.pose.position.z ) - self.goalPose.z ) +" "+ repr(euler[0])  +" "+ repr(euler[1])  +" "+ repr(euler[2]) +"\n"
        finally:
            self.lock.release()


        self.file.write(s)

    def updateGoalPose(self, landMarkMap):
        self.lock.acquire()
        try:

            for mark in landMarkMap.landmark :
                if self.landmark_id == mark.landmark_id :
                    self.goalPose = mark.position

        finally:
            self.lock.release()
    def updateRobotPose (self, odometry):
        self.lock.acquire()
        try:
            self.robotPose = odometry
        finally:
            self.lock.release()




if __name__ == '__main__':
    try:
        rospy.init_node('learning_record')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record = LearningRecord( rospy.get_name() )
        rospy.spin()
    except rospy.ROSInterruptException: pass
