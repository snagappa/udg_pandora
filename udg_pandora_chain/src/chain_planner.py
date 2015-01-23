#!/usr/bin/env python

# ROS imports
import rospy
import numpy as np
import cv2 
import math 
from cola2_lib import cola2_lib 
from auv_msgs.msg import WorldWaypointReq, NavSts
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
#from nav_msgs.msg import Odometry
import tf
import threading
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets.samples_generator import make_blobs

#from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from sklearn.cluster import AffinityPropagation

#import pylab as pl
from itertools import cycle

import ipdb

class ChainPlanner:
    def __init__(self, name):
        
        self.name = name
        
        self.min_num_det_x_cluster = 3
        self.direction = False
        self.orientation_line = 0.7
        self.rot_matrix = np.array([[np.cos(self.orientation_line), np.sin(self.orientation_line), 0],
                                     [-np.sin(self.orientation_line), np.cos(self.orientation_line), 0],
                                      [0, 0, 1]])

        self.error_threshold = 0.3
        self.iter_wps = 0
        self.cluster_centers_sorted = None    
        self.markerArray = None
        
        self.lock = threading.RLock()


        # Create Subscriber Updates (z)
        rospy.Subscriber('/link_pose2',
                         MarkerArray,
                         self.sonar_waypoint_update)
                         
                         
        rospy.Subscriber("/cola2_navigation/nav_sts", NavSts, self.updateNavSts)
        
        #Create Publisher
        self.pub_sonar_wps = rospy.Publisher("/udg_pandora/link_waypoints", MarkerArray)  
        self.pub_sonar_next_wp = rospy.Publisher("/udg_pandora/next_waypoint", Marker)
        self.pub_wwr = rospy.Publisher("/udg_pandora/world_waypoint_req", WorldWaypointReq)
        self.pub_chain_orientation = rospy.Publisher("/udg_pandora/chain_orientation", Float32)

        #Timer
        rospy.Timer(rospy.Duration(0.1), self.iterate)
        
    def iterate(self, event):

        self.lock.acquire()
        
        if self.markerArray != None:
            self.pub_sonar_wps.publish(self.markerArray)   
        
        self.lock.release()

    def updateNavSts(self, nav_sts):
        x = nav_sts.position.north
        y = nav_sts.position.east
        z = nav_sts.position.depth
        
        if self.cluster_centers_sorted != None:
            
            #set next point to go
            x_des = self.cluster_centers_sorted[self.iter_wps,0]
            y_des = self.cluster_centers_sorted[self.iter_wps,1]
            z_des = self.cluster_centers_sorted[self.iter_wps,2]
            
            ex = x - x_des
            ey = y - y_des
            ez = z - z_des
            
            #Criteria for reaching WP
            error = np.sqrt(ex**2+ey**2+0.0*ez**2)
            print "Error:", error
    
            if error < self.error_threshold and self.iter_wps < len(self.cluster_centers_sorted)-1:
                self.iter_wps = self.iter_wps + 1
                
            marker = Marker()
            marker.type = Marker.SPHERE
            marker.pose.position.x =  self.cluster_centers_sorted[self.iter_wps,0]
            marker.pose.position.y =  self.cluster_centers_sorted[self.iter_wps,1]       
            marker.pose.position.z =  self.cluster_centers_sorted[self.iter_wps,2]                             
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 0.0
            marker.header.frame_id = '/world'
            marker.header.stamp = rospy.Time.now()
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.id = self.iter_wps
            #marker.lifetime = rospy.Duration(5.0)
    
            self.pub_sonar_next_wp.publish(marker)    
            
            data = WorldWaypointReq()        
            
            data.header.stamp = rospy.Time.now()
            data.header.frame_id = "/world"
            data.goal.priority = 0
            data.goal.id = self.iter_wps
            data.altitude_mode = False   
        
            data.position.north = self.cluster_centers_sorted[self.iter_wps,0]
            data.position.east =  self.cluster_centers_sorted[self.iter_wps,1]
            data.position.depth = self.cluster_centers_sorted[self.iter_wps,2]
            data.altitude =  0.0
    
            data.orientation.roll = 0.0
            data.orientation.pitch = 0.0
            data.orientation.yaw = 0.0
      
            data.disable_axis.x = False
            data.disable_axis.y = False
            data.disable_axis.z = False
            data.disable_axis.roll = True
            data.disable_axis.pitch = True
            data.disable_axis.yaw = False
    
            data.position_tolerance.x = 0.0
            data.position_tolerance.y = 0.0
            data.position_tolerance.z = 0.0
            data.orientation_tolerance.roll = 0.0
            data.orientation_tolerance.pitch = 0.0
            data.orientation_tolerance.yaw = 0.0
            
            self.pub_wwr.publish(data)
              

    def sonar_waypoint_update(self, sonarPoints):
             
        self.numberWPS = len(sonarPoints.markers)
        #print "WPS Number:", self.numberWPS
        self.WPS = np.empty((self.numberWPS, 6), dtype=float)          
           
        for i in range (0,self.numberWPS):
            self.WPS[i][0] = sonarPoints.markers[i].pose.position.x
            self.WPS[i][1] = sonarPoints.markers[i].pose.position.y
            self.WPS[i][2] = sonarPoints.markers[i].pose.position.z
            		
            qx =  sonarPoints.markers[i].pose.orientation.x
            qy =  sonarPoints.markers[i].pose.orientation.y
            qz =  sonarPoints.markers[i].pose.orientation.z
            qw =  sonarPoints.markers[i].pose.orientation.w
            RPY = tf.transformations.euler_from_quaternion([qx, qy, qz, qw]) 
            
            self.WPS[i][3] = RPY[0]
            self.WPS[i][4] = RPY[1]
            self.WPS[i][5] = RPY[2]

            #print self.WPS[i][0], ",", self.WPS[i][1]
        print len(self.WPS)
        
        #ipdb.set_trace()
        # pass the x,y,z of the detections to the clustering algorithm
        self.cluster_meanshift(self.WPS[:,0:3]) 
        #self.cluster_dbscan(self.WPS[:,0:2])
        #self.cluster_affinity(self.WPS[:,0:2])
      
    def cluster_meanshift(self, X):
        
        cluster_centers_filtered = np.array([])
        # Compute clustering with MeanShift
        
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X[:,0:2], quantile=0.2)
        #print "bandwidth: ", bandwidth
        
        ms = MeanShift(bandwidth=0.2, bin_seeding=False, min_bin_freq=10, cluster_all=False)
        ms.fit(X[:,0:2])
        
        labels = ms.labels_
        #ipdb.set_trace()
        cluster_centers = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        #remove -1 label
        labels_unique = [x for x in labels_unique if x >= 0]
        n_clusters_ = len(labels_unique)
        
      
        
        print("number of estimated clusters : %d" % n_clusters_)
        
        self.lock.acquire()
              
        self.markerArray = MarkerArray()
        #fig = pl.figure(1)
        #pl.ion()        
        #pl.clf()
        
        cluster_centers = np.append(cluster_centers, np.zeros([len(cluster_centers),1]),1)
                   
        colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
        for k, col in zip(range(n_clusters_), colors):
               
                my_members = labels == k
                
                try: 
                    #mean of all detection z's
                    cluster_centers[k,2] = sum(X[my_members,2])/len(X[my_members,2])
                    cluster_center = cluster_centers[k]
                except: 
                    print "k: ", k, "len de cluster centers: ", len(cluster_centers)  
                
                #pl.plot(X[my_members, 0], X[my_members, 1], col + '.')
                #pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                #     markeredgecolor='k', markersize=14)
                #ipdb.set_trace()
                print "len my member: " , sum(my_members)
                if len(cluster_centers_filtered) != 0:
                        cluster_centers_filtered = np.vstack((cluster_centers_filtered,cluster_centers[k]))
                else:
                        cluster_centers_filtered = cluster_centers[k]
 
        self.sort_cluster_centers([cluster_centers_filtered, ])

        points_cluster_centers = []

        for k in range(len(self.cluster_centers_sorted)):
                #if np.sum(my_members) > self.min_num_det_x_cluster:
                marker = Marker()
                marker.type = Marker.SPHERE
                marker.pose.position.x = self.cluster_centers_sorted[k,0]
                marker.pose.position.y = self.cluster_centers_sorted[k,1]          
                marker.pose.position.z = self.cluster_centers_sorted[k,2]                                       
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 0.0
                marker.header.frame_id = '/world'
                marker.header.stamp = rospy.Time.now()
                marker.scale.x = 0.1 + k*0.05
                marker.scale.y = 0.1 + k*0.05
                marker.scale.z = 0.1 + k*0.05
                marker.color.r = 1.0
                marker.color.g = 0.0 + k*0.07
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.id = k
                self.markerArray.markers.append(marker)
                
                # Gather all points of clusters from current to last detected to compute the chain main axis line
                if k >= self.iter_wps:
                        points_cluster_centers.append([self.cluster_centers_sorted[k,0], self.cluster_centers_sorted[k,1], self.cluster_centers_sorted[k,2]])
                        print 'Waypoints per calcular la ratlla:', k
 
        # Fit line to cluster points
        vx, vy, vz, cx, cy, cz = cv2.fitLine(np.array(np.float32(points_cluster_centers)), cv2.cv.CV_DIST_HUBER, 0, 0.01, 0.01)        
        # Publish a line marker to visualize main chain axis
        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.header.frame_id = '/world'
        marker.header.stamp = rospy.Time.now()
        marker.scale.x = 0.1
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.id = 10000
	
        p1 = Point()
        p1.x = cx
        p1.y = cy
        p1.z = cz

        p2 = Point()
        p2.x = vx*4 + cx
        p2.y = vy*4 + cy
        p2.z = vz*4 + cz

        marker.points.append(p1)
        marker.points.append(p2)
        self.markerArray.markers.append(marker)     
   
        # Update orientation of main chain axis line		   
        self.orientation_line = math.atan2((p2.y-p1.y),(p2.x-p1.x))
        if self.direction:
                self.orientation_line = cola2_lib.normalizeAngle(self.orientation_line + math.pi)
        print 'Line orientation: ', math.degrees(self.orientation_line)

        self.pub_chain_orientation.publish(Float32(self.orientation_line))
        #my_members = labels == -1             
        #pl.plot(X[my_members, 0], X[my_members, 1], 'k' + '.')
            
        #pl.title('Estimated number of clusters: %d' % n_clusters_)      
        #pl.draw(i)

        self.lock.release()
  
    def sort_cluster_centers(self, cluster_centers):
      
        #multiply all cluster centers by the rotation matrix
        print "Cluster centers ", len(cluster_centers)
        cluster_centers_rot = np.zeros((len(cluster_centers[0]),3))
        
        self.rot_matrix = np.array([[np.cos(self.orientation_line), np.sin(self.orientation_line), 0],
                                     [-np.sin(self.orientation_line), np.cos(self.orientation_line), 0],
                                      [0, 0, 1]])


        for i in range(len(cluster_centers[0])):
            cluster_centers_rot[i,0:] = np.dot(self.rot_matrix,cluster_centers[0][i, :]) 
        
        #sort clusters by first coordinate     
        cluster_centers_ind_sorted = np.argsort(cluster_centers_rot[:,0])
     
        self.cluster_centers_sorted = cluster_centers[0][cluster_centers_ind_sorted,:]
	

        
          
#    def cluster_dbscan(self, X):
#        # Compute DBSCAN
#        db = DBSCAN(eps=0.5, min_samples=3, metric='euclidean').fit(X)
#        core_samples = db.core_sample_indices_
#        labels = db.labels_
#        
#        # Number of clusters in labels, ignoring noise if present.
#        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#        
#        print('Estimated number of clusters: %d' % n_clusters_)
#        print("Silhouette Coefficient: %0.3f"
#              % metrics.silhouette_score(X, labels))
#              
#        pl.figure(1)
#        pl.ion()          
#        pl.clf()      
#        
#      # Black removed and is used for noise instead.
#        unique_labels = set(labels)
#        colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#        for k, col in zip(unique_labels, colors):
#            if k == -1:
#                # Black used for noise.
#                col = 'k'
#                markersize = 6
#            class_members = [index[0] for index in np.argwhere(labels == k)]
#            cluster_core_samples = [index for index in core_samples
#                                    if labels[index] == k]
#            for index in class_members:
#                x = X[index]
#                if index in core_samples and k != -1:
#                    markersize = 14
#                else:
#                    markersize = 6
#                pl.plot(x[0], x[1], 'o', markerfacecolor=col,
#                        markeredgecolor='k', markersize=markersize)
#
#        pl.title('Estimated number of clusters: %d' % n_clusters_)
#        pl.draw()
#      
#    def cluster_affinity(self, X):
#        # Compute Affinity Propagation
#        af = AffinityPropagation(preference=None, damping=0.5).fit(X)
#        cluster_centers_indices = af.cluster_centers_indices_
#        labels = af.labels_
#        
#        n_clusters_ = len(cluster_centers_indices)
#        
#        print('Estimated number of clusters: %d' % n_clusters_)
#        print("Silhouette Coefficient: %0.3f"
#              % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
#        
#       
#        # Plot result
#       
#        pl.figure(1)
#        pl.ion()
#        pl.clf()
#        
#        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#        for k, col in zip(range(n_clusters_), colors):
#            class_members = labels == k
#            cluster_center = X[cluster_centers_indices[k]]
#            pl.plot(X[class_members, 0], X[class_members, 1], col + '.')
#            pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#                    markeredgecolor='k', markersize=14)
#            for x in X[class_members]:
#                pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#        
#        pl.title('Estimated number of clusters: %d' % n_clusters_)
#        pl.draw()

      
   
if __name__ == '__main__':
    rospy.init_node('chain_planner')
    print "Chain Planner Initialized"
    chain_planner = ChainPlanner(rospy.get_name())
    rospy.spin()
