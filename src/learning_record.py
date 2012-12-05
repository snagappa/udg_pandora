#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import std_msgs.msg
import std_srvs.srv

#include "geometry_msgs/PoseStamped.h"
from geometry_msgs.msg import PoseStamped

from tf.transformations import euler_from_quaternion

class LearningRecord:

    def __init__(self, name):
        self.name = name
        self.getConfig()
        rospy.Subscriber("/arm/pose_stamped", PoseStamped, self.updateArmPose)


    def getConfig(self):
       if rospy.has_param('learning/record/filename') :
           self.filename = rospy.get_param('learning/record/filename')
       else :
           rospy.logerr('Prameter filename not found')

       if rospy.has_param('learning/record/number_sample') :
           self.numberSample = rospy.get_param('learning/record/number_sample')
       else :
           rospy.logerr('Prameter nbDataRepro not found')
           
       self.file = open( self.filename + "_" + str(self.numberSample) +".csv", 'w')

    def updateArmPose(self, armPose):
#        euler = euler_from_quaternion( armPose.pose.orientation, 'sxyz' )  #,  axes='sxyz' );
        quaternion = [armPose.pose.orientation.x, armPose.pose.orientation.y, armPose.pose.orientation.z, armPose.pose.orientation.w ]
        euler = euler_from_quaternion( quaternion, 'sxyz' )
        s = repr(armPose.pose.position.x)+" "+ repr(armPose.pose.position.y) + " " + repr(armPose.pose.position.x) +" "+ repr(euler[0])  +" "+ repr(euler[1])  +" "+ repr(euler[2]) +"\n"
        
        self.file.write(s)

if __name__ == '__main__':
    try:
        rospy.init_node('learning_record')
#        acoustic_detectorvisual_detector = AcousticDetector(rospy.get_name())
        learning_record = LearningRecord( rospy.get_name() )
        rospy.spin() 
    except rospy.ROSInterruptException: pass
