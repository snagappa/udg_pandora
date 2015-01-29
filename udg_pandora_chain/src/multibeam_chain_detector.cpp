#include "ros/ros.h"
#include <ros/console.h>
#include <rosbag/bag.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#include "cola2_lib/cola2_util.h"
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <auv_msgs/NavSts.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <laser_geometry/laser_geometry.h>

#include <pcl_ros/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <iostream>
#include <stdlib.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <visualization_msgs/Marker.h>

class MultibeamChainDetector{

public:
	
	MultibeamChainDetector(const std::string name):
	_name(name)
	{}

	~MultibeamChainDetector() {}

	void 
	init() throw (std::runtime_error)
	{

        _pub_pointcloud = _n.advertise<sensor_msgs::PointCloud2>("/udg_pandora_chain/chain_pointcloud", 1);
        _pub_image = _n.advertise<sensor_msgs::Image>("/udg_pandora_chain/image_chain_pointcloud", 1);
        _pub_image_filtered = _n.advertise<sensor_msgs::Image>("/udg_pandora_chain/image_chain_filtered", 1);
        _pub_marker = _n.advertise<visualization_msgs::Marker>("/udg_pandora_chain/chain_detection", 1);
        _pub_pose_cs = _n.advertise<geometry_msgs::PoseWithCovarianceStamped>("/pose_ekf_slam/landmark_update/chain_pose", 1);


	    // Subscribe to multibeam laser scan
	    _sub_multibeam_scan = _n.subscribe( "/multibeam_scan", 1, &MultibeamChainDetector::updateLaserScan, this);
        // Subscribe to NavSts
        _sub_nav_sts = _n.subscribe( "/cola2_navigation/nav_sts", 1, &MultibeamChainDetector::updateNavSts, this);

        _sub_odometry = _n.subscribe( "/pose_ekf_slam/odometry", 1, &MultibeamChainDetector::updateOdometry, this);

        _buffer_size = 0;
        _max_range = 3.5;
        _resolution = 0.05;
        _chain_orientation = 0.7;
    	//Params of blob fitering
	    _params.minDistBetweenBlobs = 10.0;
	    _params.filterByColor = false;
	    _params.filterByInertia = false;
	    _params.filterByConvexity = false;
	    _params.filterByCircularity = true;
	    _params.filterByArea = true;
	    _params.minCircularity = 0.6;
	    _params.maxCircularity = 1.0;
	    _params.minArea = 9.0;
	    _params.maxArea = 64.0;
        //getConfig();
	}

    void
    updateNavSts(const auv_msgs::NavSts::ConstPtr& msg){
    
       _current_depth = msg->position.depth;
        
    }

    void
    updateOdometry(const nav_msgs::Odometry::ConstPtr& msg){

        tf::Transform robot2world;
        robot2world.setOrigin(tf::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));
        robot2world.setRotation(tf::Quaternion(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w));

        
        _world2robot = robot2world.inverse(); 

    }


	void
	updateLaserScan(const sensor_msgs::LaserScan::ConstPtr& scan){
		std::cout << "Laser Scan received " << std::endl;
    
        std::cout << "Min range: " << scan->range_min << ", Max range: " << scan->range_max << std::endl;


        if(!_listener.waitForTransform(
            scan->header.frame_id,
                    "/world",
                    scan->header.stamp + ros::Duration().fromSec(scan->ranges.size()*scan->time_increment),
                    ros::Duration(1.0))){
         return;
           }

        sensor_msgs::PointCloud2 cloud;
        _projector.transformLaserScanToPointCloud("/world",*scan, cloud, _listener, _max_range, laser_geometry::channel_option::Intensity);

        _buffer_size++;

        pcl::concatenatePointCloud( _accumulated_point_cloud, cloud, _accumulated_point_cloud);   

        if (_buffer_size > 10) {

            _buffer_size = 0;
           
            _pub_pointcloud.publish(_accumulated_point_cloud);

            cloud_conversion(_accumulated_point_cloud);

            _accumulated_point_cloud.width = 0;
            _accumulated_point_cloud.data.clear();
        }


	}

    void
    cloud_conversion(const sensor_msgs::PointCloud2 input){

            pcl::PCLPointCloud2 pcl_pc;
            
            pcl_conversions::toPCL(input, pcl_pc);

            pcl::PointCloud<pcl::PointXYZ> cloud;
            
            pcl::fromPCLPointCloud2(pcl_pc, cloud);
           
            //std::cout << cloud.at(0) << std::endl;
    

            double max_x, min_x, max_y, min_y;

            if(cloud.width * cloud.height > 0){
                max_x = cloud.at(0).x;
                min_x = cloud.at(0).x;
                max_y = cloud.at(0).y;
                min_y = cloud.at(0).y;
                      
                for(int i = 0; i < cloud.width * cloud.height; i++){

                    if(cloud.at(i).x > max_x)
                        max_x = cloud.at(i).x;
                    if(cloud.at(i).x < min_x)
                        min_x = cloud.at(i).x;
                    if(cloud.at(i).y > max_y)
                        max_y = cloud.at(i).y;
                    if(cloud.at(i).y < min_y)
                        min_y = cloud.at(i).y;                  
                    
                }
            }


            _image = cv::Mat(int((max_x-min_x)/_resolution)+1 , int((max_y-min_y)/_resolution)+1, CV_8UC1, cv::Scalar(0) );

            
            for(int i = 0; i < cloud.width * cloud.height; i++){

              //std::cout << "image size: " << _image.size().width << "," << _image.size().height << std::endl;
                int x = int((cloud.at(i).x - min_x)/_resolution);
                int y = int((cloud.at(i).y - min_y)/_resolution);
              //std::cout << "x: " << x << ", y: " << y << std::endl;
               _image.at<uchar>(x,y) = 255;
            } 

             
            cv_bridge::CvImage out_img;

            out_img.header.stamp = ros::Time::now();

            out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
            
            out_img.image = _image; 

            _pub_image.publish(out_img.toImageMsg());

            //Erode image
            unsigned int erode_size = 5;
            cv::Mat filt_img;
            cv::Mat element_erode = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(2*erode_size+1, 2*erode_size+1), cv::Point(erode_size, erode_size));
           
            //cv::erode(out_img.image, filt_img, element_erode);
            //Dilate image 
            unsigned int dilate_size = 4;
            cv::Mat element_dilate = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(2*dilate_size+1, 2*dilate_size+1), cv::Point(dilate_size, dilate_size));
          
            cv::dilate(out_img.image, filt_img, element_dilate);

            cv::erode(filt_img, out_img.image, element_erode);
           
	        out_img.header.stamp = ros::Time::now();

	        _pub_image_filtered.publish(out_img.toImageMsg());

	        //Filter image blobs
 	        cv::SimpleBlobDetector blob_detector(_params);
	        //Blob Detection
	        std::vector<cv::KeyPoint> keypoints;
	        blob_detector.detect(out_img.image, keypoints);
	        //Extract x y coordinates of the keypoint
	        for(int i=0; i<keypoints.size() ; i++){
		    
		        float X = keypoints[i].pt.x;
		        float Y = keypoints[i].pt.y;
		        std::cout << "Blob number: " << i << " X: " << X << " Y: " << Y << std::endl;
            }

            //If only one blob after the filtering we will assume is the chain
            if( keypoints.size() == 1){
                
                // Pass from image pixels to world coordinates
                float x_world = keypoints[0].pt.x*_resolution + min_x;
                float y_world = max_y - keypoints[0].pt.y*_resolution;

                // Pass the coordinates of the blob to be with respect the vehicle
                tf::Vector3 chain_position_world(x_world, y_world, 3.0); //fixed depth wrt world
                tf::Vector3 chain_position_robot = _world2robot*chain_position_world;

                tf::Quaternion q = tf::createQuaternionFromYaw(_chain_orientation);
                tf::Quaternion q_robot = _world2robot*q;

                visualization_msgs::Marker marker;
                marker.header.frame_id = "/girona500";
                marker.header.stamp = ros::Time::now();
                marker.id = 100;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.position.x = chain_position_robot[0];
                marker.pose.position.y = chain_position_robot[1];
                marker.pose.position.z = 0.0;
                marker.scale.x = 0.5;
                marker.scale.y = 0.5;
                marker.scale.z = 0.5;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 1.0;
                marker.lifetime = ros::Duration(10.0);

                _pub_marker.publish(marker);

                geometry_msgs::PoseWithCovarianceStamped pose_cs;
                pose_cs.header.stamp = ros::Time::now();
                pose_cs.header.frame_id = "/girona500";
                pose_cs.pose.covariance[0] = 0.2;
                pose_cs.pose.covariance[7] = 0.2;
                pose_cs.pose.covariance[14] = 0.1;
                pose_cs.pose.covariance[21] = 0.01;
                pose_cs.pose.covariance[28] = 0.01;
                pose_cs.pose.covariance[35] = 0.01;

                pose_cs.pose.pose.position.x = chain_position_robot[0];
                pose_cs.pose.pose.position.y = chain_position_robot[1];
                pose_cs.pose.pose.position.z = chain_position_robot[2]; 
                pose_cs.pose.pose.orientation.x = q_robot[0];
                pose_cs.pose.pose.orientation.y = q_robot[1];
                pose_cs.pose.pose.orientation.z = q_robot[2];
                pose_cs.pose.pose.orientation.w = q_robot[3];
                
                _pub_pose_cs.publish(pose_cs);
            }

    }



/*    void
    getConfig()
    {
        //Take the TF vector param and copy to a std::vector<double>
        if(!ros::param::getCached("multibeam_chain_detector/buffer_size", _config.buffer_size)) {ROS_FATAL("Invalid parameters for multibeam_chain_detector/buffer_size in param server!"); ros::shutdown();}

    }
*/
private:
	
	// ROS node
	ros::NodeHandle _n;
	ros::Subscriber _sub_multibeam_scan, _sub_nav_sts, _sub_odometry;
    ros::Publisher _pub_pointcloud, _pub_image, _pub_image_filtered, _pub_marker, _pub_pose_cs;

	// Others
	std::string _name;
    laser_geometry::LaserProjection _projector;
    tf::TransformListener _listener;
    tf::Transform _world2robot;

    double _max_range, _resolution, _current_depth, _chain_orientation;
    unsigned int _buffer_size;
    sensor_msgs::PointCloud2 _accumulated_point_cloud;

	cv::Mat _image;
	cv::SimpleBlobDetector::Params _params;
};

int 
main(int argc, char** argv) 
{
	ros::init(argc, argv, "multibeam_chain_detector");
	MultibeamChainDetector mcd(ros::this_node::getName());
			
	mcd.init();

	while (ros::ok()) {
		ros::spinOnce();
	}
	return 0;
}
