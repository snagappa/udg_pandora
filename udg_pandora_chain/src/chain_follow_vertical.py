#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROS imports
import roslib 
roslib.load_manifest('cola2_control')
import rospy
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty, EmptyRequest

# Msgs and srv imports
from cola2_control.srv import StareLandmark, StareLandmarkRequest
from cola2_control.srv import GotoWithYaw, GotoWithYawRequest
from auv_msgs.msg import NavSts, BodyVelocityReq, GoalDescriptor
from pose_ekf_slam.msg import Map


class ChainFollowVertical:

    def __init__(self,name):

        self.name = name
        self.depth = 0.0
        self.altitude = 1.0
        self.chain_detected = False

        # Publishers
        self.pub_bvr = rospy.Publisher("/cola2_control/body_velocity_req", BodyVelocityReq, queue_size = 1)
        
        # Subscribers
        rospy.Subscriber("/cola2_navigation/nav_sts", NavSts, self.update_nav_sts)
        rospy.Subscriber("/pose_ekf_slam/map", Map, self.update_map)
                            
        # Init Service Clients
        try:
            rospy.wait_for_service('/cola2_control/enable_stare_chain', 20)
            self.stare_landmark_srv = rospy.ServiceProxy(
		        '/cola2_control/enable_stare_chain', StareLandmark)
        except rospy.exceptions.ROSException:
		    rospy.logerr('%s, Error creating client.', name)
		    rospy.signal_shutdown('Error creating stare chain client')

        try:
		    rospy.wait_for_service('/cola2_control/disable_stare_chain', 20)
		    self.disable_stare_landmark_srv = rospy.ServiceProxy(
		        '/cola2_control/disable_stare_chain', Empty)
        except rospy.exceptions.ROSException:
		    rospy.logerr('%s, Error creating client.', name)
		    rospy.signal_shutdown('Error creating disable stare chain client')

        try:
            rospy.wait_for_service('/cola2_control/goto_holonomic_block', 20)
            self.goto_holonomic_srv = rospy.ServiceProxy(
                '/cola2_control/goto_holonomic_block', GotoWithYaw)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating goto holonomic block client')

        try:
            rospy.wait_for_service('/cola2_control/enable_keep_position_g500', 20)
            self.enable_keep_position_srv = rospy.ServiceProxy(
                '/cola2_control/enable_keep_position_g500', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating enable keep position client')

        try:
            rospy.wait_for_service('/cola2_control/disable_keep_position', 20)
            self.disable_keep_position_srv = rospy.ServiceProxy(
                '/cola2_control/disable_keep_position', Empty)
        except rospy.exceptions.ROSException:
            rospy.logerr('%s, Error creating client.', name)
            rospy.signal_shutdown('Error creating disable keep position client')

    def up_and_down(self, max_depth, min_depth, min_altitude):

        body_velocity_req = BodyVelocityReq()
        # header & goal
        
        body_velocity_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        body_velocity_req.goal.requester = self.name + '_velocity'

        body_velocity_req.twist.linear.z = 0.05

        # Check if DoF is disable
        body_velocity_req.disable_axis.x = True
        body_velocity_req.disable_axis.y = True
        body_velocity_req.disable_axis.z = False
        body_velocity_req.disable_axis.roll = True
        body_velocity_req.disable_axis.pitch = True
        body_velocity_req.disable_axis.yaw = True

        r = rospy.Rate(10)

        while (self.depth < max_depth) and (self.altitude > min_altitude) :
            body_velocity_req.header.stamp = rospy.Time().now()
            self.pub_bvr.publish(body_velocity_req)
            r.sleep()
        
        body_velocity_req.twist.linear.z = -0.05
        
        while (self.depth > min_depth):
            body_velocity_req.header.stamp = rospy.Time().now()
            self.pub_bvr.publish(body_velocity_req)
            r.sleep()

    def update_map(self,data):
        #search for landmark of id chain_pose
        for i in range(len(data.landmark)):
            if data.landmark[i].landmark_id == "/pose_ekf_slam/landmark_update/chain_pose":
                self.chain_detected = True
            else:
                self.chain_detected = False
        
    def update_nav_sts(self,data):    
        self.depth = data.position.depth
        self.altitude = data.altitude


    def stare(self, distance, block):
        req = StareLandmarkRequest()

        req.landmark_id = "/pose_ekf_slam/landmark_update/chain_pose" 
        req.offset.position.x = 0.0 
        req.offset.position.y = distance
        req.offset.position.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, -1.57)

        req.offset.orientation.x = quat[0]
        req.offset.orientation.y = quat[1]
        req.offset.orientation.z = quat[2]
        req.offset.orientation.w = quat[3]

        req.tolerance = 0.25
        req.keep_pose = block

        self.stare_landmark_srv(req)

    def inspect_chain(self):
        self.stare(2.0, True)
        self.disable_stare_landmark_srv(EmptyRequest())
        self.stare(2.0, False)
        self.up_and_down(4.5, 1.0, 1.0)
        self.disable_stare_landmark_srv(EmptyRequest())

    def search_chain(self, waypoint_list, wait_time, search_depth):
        
        req = GotoWithYawRequest()
        req.z = search_depth
        req.altitude_mode = False
        req.tolerance = 0.2
        
        index_wp = 0
         
        while self.chain_detected == False and index_wp<len(waypoint_list):
            
            # Goto waypoint
            req.north_lat = waypoint_list[index_wp][0]
            req.east_lon = waypoint_list[index_wp][1]
            req.yaw = waypoint_list[index_wp][2]
            print "Going to WP: x:", waypoint_list[index_wp][0], ", y: ", waypoint_list[index_wp][1], ", yaw: ", waypoint_list[index_wp][2]
            self.goto_holonomic_srv(req)
            print "WP ", index_wp, "reached"

            #Keep position during 5 seconds to allow time for chain detection
            self.enable_keep_position_srv(EmptyRequest())
            
            rospy.sleep(wait_time)
            #Disable keep position
            self.disable_keep_position_srv(EmptyRequest())
            
            index_wp = index_wp + 1 

        if self.chain_detected 
            print "CHAIN FOUND"

        return self.chain_detected

    def main_mission(self):
        
        found = self.search_chain([[1.,1.,0.],[1.,1.,-1.57],[1.,1.,0.]],5.0, 2.5)

        if found:
            self.inspect_chain()
        else:
            print "Chain not found during the search pattern"

if __name__ == '__main__':
        
        try:
            name = 'chain_follow_vertical_req'
            rospy.init_node(name)
    
            chain_follow_vertical = ChainFollowVertical(rospy.get_name())
            chain_follow_vertical.main_mission()
            
        except rospy.ROSInterruptException: 
            pass





        
