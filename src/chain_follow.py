#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

from auv_msgs.msg import WorldWaypointReq
from auv_msgs.msg import BodyVelocityReq
from visualization_msgs.msg import MarkerArray
import tf
from cola2_perception_dev.msg import SonarInfo
import numpy as np
from auv_msgs.msg import GoalDescriptor

class ChainFollow:
    def __init__(self, name):
        self.name = name

        # Default parameters
        self.window_lenght = 3.0
        self.window_start = 1.0

        # self.get_config()

        self.listener = tf.TransformListener()

        self.pub_yaw_rate = rospy.Publisher('/cola2_control/body_velocity_req',
                                            BodyVelocityReq)
        self.pub_waypoint_req = rospy.Publisher('/cola2_control/world_waypoint_req',
                                                WorldWaypointReq)

        # Create Subscriber Updates (z)
        rospy.Subscriber('/ntua_planner/sonar_waypoints',
                         MarkerArray,
                         self.sonar_waypoint_update)

        rospy.Subscriber('/cola2_perception/soundmetrics_aris300/sonar_info',
                         SonarInfo,
                         self.sonar_info_update)


    def sonar_info_update(self, data):
        self.window_lenght = data.window_lenght
        self.window_start = data.window_start


    def sonar_waypoint_update(self, data):
        self.listener.waitForTransform( 
                                       '/world', '/soundmetrics_aris3000_img',
                                       rospy.Time(),
                                       rospy.Duration(2.0))
                        
        (sensor_trans, sensor_rot) = self.listener.lookupTransform(
                                       
                                       '/world','/soundmetrics_aris3000_img',
                                       rospy.Time(0))

        self.listener.waitForTransform( 
                                       '/world', '/girona500',
                                       rospy.Time(),
                                       rospy.Duration(2.0))
                        
        (trans_g500, rot_g500) = self.listener.lookupTransform(
                            		
                            		'/world','/girona500',
                            		rospy.Time(0))
        print 'g500 rot: \n', rot_g500
        print 'sensor trans: \n', sensor_trans
        print 'g500 trans: \n', trans_g500
        print 'sensor rot: \n', sensor_rot
        # Transform all waypoint from world frame to sensor frame
        wTs = tf.transformations.quaternion_matrix(rot_g500)
        wTs[0:3, 3] = sensor_trans
        sTw = np.linalg.pinv(wTs)
        list_of_wp = list()
        for waypoint in data.markers:
            Pw = np.array([0, 0, 0, 1.0])
            Pw[0] = waypoint.pose.position.x
            Pw[1] = waypoint.pose.position.y
            Pw[2] = waypoint.pose.position.z
            Ps = np.dot(sTw, Pw)
            list_of_wp.append(Ps)

        # take those points inside field of view
        # Rectangle that approximates FOV
        # cross range = total range / 2 (Tali says)
        min_x = -self.window_lenght/2.0
        max_x = self.window_lenght/2.0
        min_y = -(self.window_lenght + self.window_start)/4.0
        max_y = (self.window_lenght + self.window_start)/4.0
        max_wp_x = -999
        max_wp_index = -1
        i = 0
        for waypoint in list_of_wp:
            print '--> ', waypoint
            if waypoint[0] > min_x and waypoint[0] < max_x and waypoint[1] > min_y and waypoint[1] < max_y:
                if waypoint[0] > max_wp_x:
                    max_wp_x = waypoint[0]
                    max_wp_index = i
            i = i + 1

        if max_wp_index >= 0:
            # It is the waypoint in the sonar fov with larger x
            self.__compute_yaw_rate__(list_of_wp[max_wp_index][1])
        else:
            rospy.loginfo("%s: No waypoints inside sonar FOV")


    def __compute_yaw_rate__(self, y_offset):
         # Publish Body Velocity Request
        body_velocity_req = BodyVelocityReq()

        # header & goal
        body_velocity_req.header.stamp = rospy.Time().now()
        body_velocity_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        body_velocity_req.goal.requester = self.name + '_velocity'

        # twist set-point
        body_velocity_req.twist.angular.z = y_offset/2.0

        # Check if DoF is disable
        body_velocity_req.disable_axis.x = True
        body_velocity_req.disable_axis.y = True
        body_velocity_req.disable_axis.z = True
        body_velocity_req.disable_axis.roll = True
        body_velocity_req.disable_axis.pitch = True
        body_velocity_req.disable_axis.yaw = False
        print 'Answer: \n', body_velocity_req	
        self.pub_yaw_rate.publish(body_velocity_req)


if __name__ == '__main__':
    rospy.init_node('chain_follow')
    chain_follow = ChainFollow(rospy.get_name())
    rospy.spin()
