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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import threading
from visualization_msgs.msg import Marker
import cola2_lib
import threading

class ChainFollow:
    def __init__(self, name):
        self.name = name

        # Default parameters
        self.window_length = 3.0
        self.window_start = 1.0
        self.waypoint_req = WorldWaypointReq()
        self.body_velocity_req = BodyVelocityReq()
        self.odometry = Odometry()
        self.sonar_img_pose = PoseStamped()
        self.lock = threading.RLock()
        self.last_waypoint = WorldWaypointReq()
        self.look_around = False
        self.yaw_offset = 0.35 # 25 degrees
        self.do_turn_around = False 
        self.lock = threading.RLock()
        self.listener = tf.TransformListener()        
        self.odometry_updated = False 
        self.big_turn_around = False
        # self.get_config()

        self.pub_yaw_rate = rospy.Publisher('/cola2_control/body_velocity_req',
                                            BodyVelocityReq)
        self.pub_waypoint_req = rospy.Publisher('/cola2_control/world_waypoint_req',
                                                WorldWaypointReq)
        self.pub_marker = rospy.Publisher('/udg_pandora/orientation_link', Marker)

        # Create Subscriber Updates (z)
        rospy.Subscriber('/ntua_planner/sonar_waypoints',
                         MarkerArray,
                         self.sonar_waypoint_update)

        rospy.Subscriber('/cola2_perception/soundmetrics_aris3000/sonar_info',
                         SonarInfo,
                         self.sonar_info_update)

        rospy.Subscriber('/ntua_planner/world_waypoint_req',
                         WorldWaypointReq,
                         self.world_waypoint_req_update)

        rospy.Subscriber('/pose_ekf_slam/odometry',
                         Odometry,
                         self.odometry_update)
        
        rospy.Subscriber('/cola2_perception/soundmetrics_aris3000/sonar_img_pose',
                         PoseStamped,
                         self.sonar_img_pose_update)

        rospy.Timer(rospy.Duration(0.05), 
                    self.publish_control)

        rospy.Timer(rospy.Duration(0.5), 
                    self.update_sonar_img_tf) 

    def update_sonar_img_tf(self, data):
        (self.trans, self.rot) = self.listener.lookupTransform('/world', '/soundmetrics_aris3000_img', rospy.Time(0))
        # print 'Trans: ', self.trans

    def odometry_update(self, data):
        self.odometry = data
        self.odometry_updated = True

    def sonar_img_pose_update(self, data):
        self.sonar_img_pose = data

    def world_waypoint_req_update(self, data):
        self.lock.acquire()
        self.waypoint_req = data
        self.waypoint_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        self.waypoint_req.goal.requester = self.name + '_pose'
        self.waypoint_req.disable_axis.z = True
        self.waypoint_req.disable_axis.yaw = True
        self.waypoint_req.disable_axis.x = False
        self.waypoint_req.disable_axis.y = False
        self.lock.release()
       
        if self.last_waypoint.goal.id != self.waypoint_req.goal.id:
            # Change waypoint
            if self.do_turn_around:
                self.last_waypoint = self.waypoint_req
                if self.big_turn_around:
                    print 'Big turn around'
                    self.look_around_movement(2)
                    self.big_turn_around = False
                else:
                    print 'Normal turn around'
                    self.look_around_movement(1)
                self.do_turn_around = False


    def sonar_info_update(self, data):
        self.window_length = data.window_length
        self.window_start = data.window_start


    def sonar_waypoint_update(self, data):
       # Transform all waypoint from world frame to sensor frame
        wTs = tf.transformations.quaternion_matrix(
                        [self.odometry.pose.pose.orientation.x,
                         self.odometry.pose.pose.orientation.y,
                         self.odometry.pose.pose.orientation.z,
                         self.odometry.pose.pose.orientation.w])
        wTs[0:3, 3] = [self.sonar_img_pose.pose.position.x,
                       self.sonar_img_pose.pose.position.y,
                       self.sonar_img_pose.pose.position.z]
        # print 'robot orientation: \n',self.odometry.pose.pose.orientation
        # print 'sonar position: \n', self.sonar_img_pose.pose.position 
        sTw = np.linalg.pinv(wTs)
        list_of_wp = list()
        for waypoint in data.markers:
            Pw = np.array([0, 0, 0, 1.0])
            Pw[0] = waypoint.pose.position.x
            Pw[1] = waypoint.pose.position.y
            Pw[2] = waypoint.pose.position.z
            Ps = np.dot(sTw, Pw)
            list_of_wp.append(Ps)

	# print 'WP wrt SONAR: ', list_of_wp

        # take those points inside field of view
        # Rectangle that approximates FOV
        # cross range = total range / 2 (Tali says)
        min_x = -self.window_length/2.0
        max_x = self.window_length/2.0
        min_y = -(self.window_length + self.window_start)/4.0
        max_y = (self.window_length + self.window_start)/4.0
        max_wp_x = -999
        max_wp_index = -1
        i = 0
        for waypoint in list_of_wp:
            # print '--> ', waypoint
            if waypoint[0] > min_x and waypoint[0] < max_x and waypoint[1] > min_y and waypoint[1] < max_y:
                if waypoint[0] > max_wp_x:
                    max_wp_x = waypoint[0]
                    max_wp_index = i
            i = i + 1

        if max_wp_index >= 0:
            # It is the waypoint in the sonar fov with larger x
            rospy.loginfo("%s: Compute yaw rate", self.name) 
            self.body_velocity_req = self.__compute_yaw_rate__(list_of_wp[max_wp_index][1])
            print 'MAX X: ', list_of_wp[max_wp_index][0], '/', -(self.window_length/4.0)
            if list_of_wp[max_wp_index][0] < -(self.window_length/4.0):
                self.do_turn_around = True           
 
            point = np.dot(wTs, list_of_wp[max_wp_index])

            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = '/world'
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.id = 66
            marker.type = Marker.SPHERE
            self.pub_marker.publish(marker)
        else:
            rospy.loginfo("%s: No waypoints inside sonar FOV", self.name)
            self.do_turn_around = True
            self.big_turn_around = True
            # if not self.look_around:
            #    self.look_around_movement(2) 

    def publish_control(self, event):
        if not self.look_around:
            self.lock.acquire()
            self.pub_yaw_rate.publish(self.body_velocity_req)
            # print self.name, ', YAW RATE: ', self.body_velocity_req.twist.angular.z
           
            if abs(self.body_velocity_req.twist.angular.z) < 0.05 and self.body_velocity_req.twist.angular.z != 0.0:
                # print self.name, ', WP: ', self.waypoint_req.position.north, ', ', self.waypoint_req.position.east
                self.waypoint_req.header.stamp = rospy.Time.now()
                self.pub_waypoint_req.publish(self.waypoint_req)
            self.lock.release()


    def look_around_movement(self, factor = 1):
        self.look_around = True
        print 'look around!'
        current_orientation = tf.transformations.euler_from_quaternion([
                                      self.odometry.pose.pose.orientation.x,
                                      self.odometry.pose.pose.orientation.y,
                                      self.odometry.pose.pose.orientation.z,
                                      self.odometry.pose.pose.orientation.w])

        waypoint_req = WorldWaypointReq()
        waypoint_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        waypoint_req.goal.requester = self.name + '_pose'
     
        waypoint_req.disable_axis.x = not self.odometry_updated
        waypoint_req.disable_axis.y = not self.odometry_updated
        waypoint_req.disable_axis.z = True
        waypoint_req.disable_axis.roll = True
        waypoint_req.disable_axis.pitch = True
        waypoint_req.disable_axis.yaw = False

        waypoint_req.position.north = float(self.odometry.pose.pose.position.x)
        waypoint_req.position.east = float(self.odometry.pose.pose.position.y)
        waypoint_req.orientation.yaw = current_orientation[2] + (factor * self.yaw_offset)
        
        for i in range(int(100*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1)
        
        waypoint_req.orientation.yaw = current_orientation[2] - (factor * self.yaw_offset)
        for i in range(int(150*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1)
 
        waypoint_req.orientation.yaw = current_orientation[2]
        for i in range(int(50*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1) 
   
        self.look_around = False
        

    def __compute_yaw_rate__(self, y_offset):
         # Publish Body Velocity Request
        body_velocity_req = BodyVelocityReq()

        # header & goal
        body_velocity_req.header.stamp = rospy.Time().now()
        body_velocity_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        body_velocity_req.goal.requester = self.name + '_velocity'

        # twist set-point
        body_velocity_req.twist.angular.z = y_offset/4.0

        # Check if DoF is disable
        body_velocity_req.disable_axis.x = True
        body_velocity_req.disable_axis.y = True
        body_velocity_req.disable_axis.z = True
        body_velocity_req.disable_axis.roll = True
        body_velocity_req.disable_axis.pitch = True
        body_velocity_req.disable_axis.yaw = False
        return body_velocity_req	
        

if __name__ == '__main__':
    rospy.init_node('chain_follow')
    chain_follow = ChainFollow(rospy.get_name())
    rospy.spin()
