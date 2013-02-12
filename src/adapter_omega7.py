#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy


from udg_pandora.msg import omega7State

# import the message to know the position
from geometry_msgs.msg import PoseStamped

#include message of the point
from sensor_msgs.msg import Joy

class AdapterOmega7 :


    def __init__(self,name):
        self.name = name
        self.getConfig()
        rospy.loginfo('Configuration Loaded')
        rospy.Subscriber(self.publisher_name, omega7State  , self.update_omega7 )

        self.pub_arm_command = rospy.Publisher("/cola2_control/joystick_arm_data", Joy )


    def getConfig(self):
        param_dict = {'publisher_name': 'adapter_omega7/publisher_name',
                      'maximum_dof': 'adapter_omega7/maximum_dof',
                      'minimum_dof': 'adapter_omega7/minimum_dof',
                      'max_joy': 'adapter_omega7/max_joy',
                      'min_joy': 'adapter_omega7/min_joy'}
        cola2_ros_lib.getRosParams(self, param_dict)
        rospy.loginfo('Interval time value: ' + str(self.interval_time) )

    # receive the omega message and convert to joy message publish to teleoperation arm
    def update_omega7(self, omega7_state):
        joy_message = Joy()
        for i in xrange( 6 ) :
            # limit [A..B] origin limit [C..D] target
            # x' = ( (D-C)*(x-A) / (B-A) ) + C
            normalized =( ( (self.max_joy[i] - self.min_joy[i])*( omega7_state.pose.position[i] - self.maximum_dof[i] ) ) / ( self.maximum_dof[i] - self.minimum_dof[i]) ) + self.min_joy[i]
            joy_message.append(normalized)

        self.pub_arm_command.publish(joy_message)





if __name__== '__main__' :
    try:
        rospy.init_node('adapter_omega7')
        adapterOmega7 =  AdapterOmega7( rospy.get_name() )
        rospy.spin()

        except rospy.ROSInterruptException: pass
