#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROS imports

import rospy
# from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty, EmptyRequest

# Msgs and srv imports
#from cola2_control.srv import GotoWithYaw, GotoWithYawRequest
from auv_msgs.msg import NavSts, BodyVelocityReq, GoalDescriptor, WorldWaypointReq
from visualization_msgs.msg import MarkerArray


class ChainFollowHorizontal:

    def __init__(self,name):
        self.name = name
        self.link_detections = [0.0]*2
        self.link_detections_index = 0
        self.total_link_wp = 0
        self.nav = NavSts() 

        # Publishers
        self.pub_wwr = rospy.Publisher("/cola2_control/world_waypoint_req", WorldWaypointReq, queue_size = 1)
        
        # Subscribers
        rospy.Subscriber("/link_pose2", MarkerArray, self.update_link_detection) 
        rospy.Subscriber("/udg_pandora/link_waypoints", MarkerArray, self.update_link_wp) 
        rospy.Subscriber('/cola2_navigation/nav_sts', NavSts, self.nav_sts_update)


    def nav_sts_update(self, msg):
        self.nav = msg


    def update_link_detection(self, msg):
        self.link_detections[self.link_detections_index%2] = msg.header.stamp.to_sec()
        self.link_detections_index = self.link_detections_index + 1


    def update_link_wp(self, msg):
        self.total_link_wp = len(msg.markers)


    def iterate(self, wp_list):
        wp = 0
        found = False
        while wp < len(wp_list) and not found:
            print 'Go to waypoint ', wp

            # Create msg for WP 'wp'
            waypoint_req = WorldWaypointReq()
            waypoint_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
            waypoint_req.goal.requester = self.name + '_pose'
            waypoint_req.disable_axis.x = False
            waypoint_req.disable_axis.y = False
            waypoint_req.disable_axis.z = False
            waypoint_req.disable_axis.roll = True
            waypoint_req.disable_axis.pitch = True
            waypoint_req.disable_axis.yaw = False
            waypoint_req.altitude_mode = False
            waypoint_req.position.north = wp_list[wp][0]
            waypoint_req.position.east = wp_list[wp][1]
            waypoint_req.position.depth = wp_list[wp][2] 
            waypoint_req.orientation.yaw = wp_list[wp][3]

            r = rospy.Rate(10)
            while not check_tolerance(wp_list[wp], 1.0) and not found:
                # Send waypoint req
                waypoint_req.header.stamp = rospy.Time.now()
                self.pub_wwr.publish(waypoint_req)

                # Check if links detected
                if self.link_detected():
                    if self.total_link_wp > 2:
                        found = True
                    else:
                        rospy.sleep(3.0)

                r.sleep()
            wp = wp + 1

        if found:
            print 'Chain Found!!!'
        else:
            print 'Chain not found!!! Waypoint list empty.'            


    def check_tolerance(self, wp, tolerance):
        if ((abs(self.nav.position.north - wp[0]) < tolerance) and
            (abs(self.nav.position.east - wp[1]) < tolerance) and
            (abs(self.nav.position.depth - wp[2]) < tolerance) and
            (abs(self.nav.orientation.yaw - wp[3]) < tolerance)):
            return True
        else:
            return False
                

    def link_detected(self):
        now = rospy.Time.now().to_sec()
        for i in len(self.link_detections):
            if now - i > 3.0:
                return False
        return True
 
if __name__ == '__main__':
        
        try:
            name = 'chain_follow_horizontal'
            rospy.init_node(name)
    
            CFH = ChainFollowHorizontal(rospy.get_name())
            CFH.main_mission()
            
        except rospy.ROSInterruptException: 
            pass



