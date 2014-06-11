#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on December 3 2013
@author: narcis palomeras
"""
# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty, EmptyRequest
from threading import Thread
from auv_msgs.msg import BodyForceReq
from auv_msgs.msg import BodyVelocityReq
from auv_msgs.msg import GoalDescriptor


class PushAndRotate(object):
    def __init__(self, name):
        self.push_init = False
        self.name = name
        self.pub_body_force_req = rospy.Publisher(
            "/cola2_control/body_force_req",
            BodyForceReq)

        self.pub_body_velocity_req = rospy.Publisher(
            "/cola2_control/body_velocity_req",
            BodyVelocityReq)

    def enable_push(self, force):
        print 'enable_push: ', force
        if not self.push_init:
            self.push_init = True
            t = Thread(target=self.enable_push_thread, args=([force]))
            t.daemon = True
            t.start()

    def disable_push(self):
        self.push_init = False
        rospy.sleep(1.0)


    def enable_push_thread(self, f):
        print 'thread: ', f
        force = BodyForceReq()
        force.header.frame_id = 'girona500'
        force.goal.requester = self.name + '_force'
        force.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        force.wrench.force.x = f
        force.disable_axis.x = False
        force.disable_axis.y = True
        force.disable_axis.z = True
        force.disable_axis.roll = True
        force.disable_axis.pitch = True
        force.disable_axis.yaw = True

        vel = BodyVelocityReq()
        vel.header.frame_id = 'girona500'
        vel.goal.requester = self.name + '_vel'
        vel.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        vel.disable_axis.x = True
        vel.disable_axis.y = False
        vel.disable_axis.z = False
        vel.disable_axis.roll = False
        vel.disable_axis.pitch = False
        vel.disable_axis.yaw = False

        rate = rospy.Rate(10)
        while self.push_init:
            print 'send ', f
            force.header.stamp = rospy.Time().now()
            vel.header.stamp = force.header.stamp

            self.pub_body_force_req.publish(force)
            self.pub_body_velocity_req.publish(vel)

            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('push_and_rotate')
        push_and_rotate = PushAndRotate(rospy.get_name())
        push_and_rotate.enable_push(20.0)
        rospy.sleep(10.0)
        push_and_rotate.disable_push()
        push_and_rotate.enable_push(-20.0)
        rospy.sleep(10.0)
        push_and_rotate.disable_push()
    except rospy.ROSInterruptException:
        pass

