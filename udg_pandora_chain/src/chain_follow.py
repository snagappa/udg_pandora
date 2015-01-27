#!/usr/bin/env python

# ROS imports
import rospy
import math
from geometry_msgs.msg import Point
from auv_msgs.msg import WorldWaypointReq
from auv_msgs.msg import BodyVelocityReq
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray
import tf
from cola2_perception.msg import SonarInfo
import numpy as np
from auv_msgs.msg import GoalDescriptor
from auv_msgs.msg import NavSts
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import threading
from visualization_msgs.msg import Marker
from cola2_lib import cola2_lib
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
        self.yaw_offset = 0.45 # 0.35 => 25 degrees
        self.do_turn_around = False 
        self.lock = threading.RLock()
        self.listener = tf.TransformListener()        
        self.broadcaster = tf.TransformBroadcaster()
        self.odometry_updated = False 
        self.big_turn_around = False
        self.chain_orientation = 0.0
        self.current_yaw = 0.0

        # self.get_config()

        self.pub_yaw_rate = rospy.Publisher('/cola2_control/body_velocity_req',
                                            BodyVelocityReq)
        self.pub_waypoint_req = rospy.Publisher('/cola2_control/world_waypoint_req',
                                                WorldWaypointReq)
        self.pub_marker = rospy.Publisher('/udg_pandora/orientation_link', Marker)

        # Create Subscriber Updates (z)
        rospy.Subscriber('/udg_pandora/link_waypoints',
                         MarkerArray,
                         self.sonar_waypoint_update,
                         queue_size = 1)

        rospy.Subscriber('/cola2_perception/soundmetrics_aris3000/sonar_info',
                         SonarInfo,
                         self.sonar_info_update,
                         queue_size = 1)

        rospy.Subscriber('/udg_pandora/world_waypoint_req',
                         WorldWaypointReq,
                         self.world_waypoint_req_update,
                         queue_size = 1)

        rospy.Subscriber('/pose_ekf_slam/odometry',
                         Odometry,
                         self.odometry_update,
                         queue_size = 1)
        
        rospy.Subscriber('/cola2_perception/soundmetrics_aris3000/sonar_img_pose',
                         PoseStamped,
                         self.sonar_img_pose_update,
                         queue_size = 1)

        rospy.Subscriber('/udg_pandora/chain_orientation',
                        Float32,
                        self.chain_orientation_update)

        rospy.Subscriber('/cola2_navigation/nav_sts',
                        NavSts,
                        self.nav_sts_update)

        rospy.Timer(rospy.Duration(0.05), 
                    self.publish_control)

        rospy.Timer(rospy.Duration(0.5), 
                    self.update_sonar_img_tf) 

    def nav_sts_update(self,data):
        self.current_yaw = data.orientation.yaw

    def chain_orientation_update(self, data):
        self.chain_orientation = data.data

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
            print 'Change WP'
            self.last_waypoint = self.waypoint_req
            # Change waypoint
            if self.do_turn_around:
                print 'Turn around is true'
                if self.big_turn_around:
                    print 'Big turn around'
                    self.look_around_movement(1)
                    self.big_turn_around = False
                else:
                    print 'Normal turn around'
                    self.look_around_movement(1)
                self.do_turn_around = False
            else:
                print 'Turn around is False'

    def sonar_info_update(self, data):
        self.window_length = data.window_length
        self.window_start = data.window_start


    def sonar_waypoint_update(self, data):
        print 'TOTAL SIZE: ', len(data.markers)

        # Transform all waypoint from world frame to sensor frame
        wTs = tf.transformations.quaternion_matrix(
                        [self.odometry.pose.pose.orientation.x,
                         self.odometry.pose.pose.orientation.y,
                         self.odometry.pose.pose.orientation.z,
                         self.odometry.pose.pose.orientation.w])
        wTs[0:3, 3] = [self.sonar_img_pose.pose.position.x,
                       self.sonar_img_pose.pose.position.y,
                       self.sonar_img_pose.pose.position.z]

        self.broadcaster.sendTransform((self.sonar_img_pose.pose.position.x, self.sonar_img_pose.pose.position.y, self.sonar_img_pose.pose.position.z),
            (self.odometry.pose.pose.orientation.x, self.odometry.pose.pose.orientation.y,self.odometry.pose.pose.orientation.z, self.odometry.pose.pose.orientation.w),
            rospy.Time.now(),
            '/sonar_tf', 
            '/world'
    	)     

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
            print Ps
        print '-------------------------------' 
	    #print 'WP wrt SONAR: ', list_of_wp

        # take those points inside field of view
        # Rectangle that approximates FOV
        # cross range = total range / 2 (Tali says)
        min_x = -self.window_length/2.0
        max_x = self.window_length/2.0
        min_y = -(self.window_length)/4.0
        max_y = (self.window_length)/4.0
        max_wp_x = -999
        max_wp_index = -1
        i = 0

        marker = Marker
        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.header.frame_id = '/sonar_tf'
        marker.header.stamp = rospy.Time.now()
        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.id = 731
        marker.lifetime = rospy.Duration(0.1)
        marker.action = Marker.ADD
        
        """p1 = Point(-min_x,-min_y, self.sonar_img_pose.pose.position.z )
        p2 = Point(-min_x,max_y, self.sonar_img_pose.pose.position.z)
        p3 = Point(max_x,max_y, self.sonar_img_pose.pose.position.z)
        p4 = Point(max_x,-min_y, self.sonar_img_pose.pose.position.z)
        p5 = Point(-min_x,-min_y, self.sonar_img_pose.pose.position.z)"""
 
        marker.points.append(Point(-min_x,-min_y, 0.0 ))
        marker.points.append(Point(-min_x,min_y, 0.0 ))
        marker.points.append(Point(min_x,min_y, 0.0 ))
        marker.points.append(Point(min_x,-min_y, 0.0 )) 
        marker.points.append(Point(-min_x,-min_y, 0.0 ))

        self.pub_marker.publish(marker)

        for waypoint in list_of_wp:
            # print '--> ', waypoint
            if waypoint[0] > min_x and waypoint[0] < max_x and waypoint[1] > min_y and waypoint[1] < max_y:
                if waypoint[0] > max_wp_x:
                    max_wp_x = waypoint[0]
                    max_wp_index = i

                # MARKER ARRAY
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = '/sonar_tf'
                marker.pose.position.x = waypoint[0]
                marker.pose.position.y = waypoint[1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.id = i*3
                marker.lifetime = rospy.Duration(0.1)
                marker.type = Marker.SPHERE
                self.pub_marker.publish(marker)
 
            i = i + 1

        if max_wp_index >= 0:
            # It is the waypoint in the sonar fov with larger x
            # rospy.loginfo("%s: Compute yaw rate", self.name) 
            self.body_velocity_req = self.__compute_yaw_rate__(list_of_wp[max_wp_index][1], list_of_wp[max_wp_index][0])
            #print 'MAX X: ', list_of_wp[max_wp_index][0], '/', -(self.window_length/4.0)
            #if list_of_wp[max_wp_index][0] < -(self.window_length/4.0):
            #    self.do_turn_around = True
            #else:
            #    self.do_turn_around = False           
 
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
            marker.scale.x = 0.3
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.id = 66
            marker.type = Marker.ARROW
            self.pub_marker.publish(marker)
        else:
            if not self.look_around:
                rospy.loginfo("%s: No waypoints inside sonar FOV", self.name)
                self.do_turn_around = True
                self.big_turn_around = True
            # if not self.look_around:
            #    self.look_around_movement(2) 
        # print 'do turn around: ', self.do_turn_around
        # print 'big turn: ', self.big_turn_around

    def publish_control(self, event):
        #print 'Publish_control: look_around is ', self.look_around 
        if not self.look_around:
            
            self.lock.acquire()
            
            self.body_velocity_req.header.stamp = rospy.Time.now()
            self.pub_yaw_rate.publish(self.body_velocity_req)
            #print self.name, ', YAW RATE: ', self.body_velocity_req.twist.angular.z
           
            if abs(self.body_velocity_req.twist.angular.z) < 0.1:
                #print self.name, ', WP: ', self.waypoint_req.goal.id 
                #print 'Distance: ', np.sqrt((self.odometry.pose.pose.position.x - self.waypoint_req.position.north)**2 + 
                #                              (self.odometry.pose.pose.position.y - self.waypoint_req.position.east)**2)

                self.waypoint_req.header.stamp = rospy.Time.now()
                self.pub_waypoint_req.publish(self.waypoint_req)
            self.lock.release()


    def look_around_movement(self, factor = 1):
        self.look_around = True
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
        waypoint_req.orientation.yaw = cola2_lib.normalizeAngle(current_orientation[2] + (factor * self.yaw_offset))
        
        for i in range(int(100*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1)
            print 'look around.'

        waypoint_req.orientation.yaw = cola2_lib.normalizeAngle(current_orientation[2] - (factor * self.yaw_offset))
        for i in range(int(150*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1)
            print 'look around..'

        waypoint_req.orientation.yaw = current_orientation[2]
        for i in range(int(50*factor)):
            waypoint_req.header.stamp = rospy.Time.now()        
            self.pub_waypoint_req.publish(waypoint_req)
            rospy.sleep(0.1) 
            print 'look around...'

        self.look_around = False
        

    def __compute_yaw_rate__(self, y_offset, x_offset):
         # Publish Body Velocity Request
        body_velocity_req = BodyVelocityReq()
        
        distance = math.sqrt(y_offset**2 + x_offset**2)

        #print 'distance compute yaw: ', distance

        if  distance > 1.5:
            # twist set-point
            body_velocity_req.twist.angular.z = y_offset/4.0
            #print '>>>>>>>>>>>>>>>>>>>>>>>bigger than 1.5 m'
        else:
            body_velocity_req.twist.angular.z = cola2_lib.normalizeAngle(self.chain_orientation - self.current_yaw)/10.0
            #print '>>>>>>>>>>>>>>>>>>>>>>>smaller than 1.5 m'
            #print 'Chain orientation:', self.chain_orientation, ' Current yaw: ', self.current_yaw

        if body_velocity_req.twist.angular.z > 0.15:
            body_velocity_req.twist.angular.z = 0.15
        elif body_velocity_req.twist.angular.z < -0.15:
            body_velocity_req.twist.angular.z = -0.15

        # header & goal
        body_velocity_req.header.stamp = rospy.Time().now()
        body_velocity_req.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        body_velocity_req.goal.requester = self.name + '_velocity'

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
