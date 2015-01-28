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

	// Subscribe to multibeam laser scan
	_sub_multibeam_scan = _n.subscribe( "/multibeam_scan", 1, &MultibeamChainDetector::updateLaserScan, this);

        _buffer_size = 0;
        _max_range = 3.0;
        _resolution = 0.05;
	//Params of blob fitering
	_params.minDistBetweenBlobs = 10.0;
	_params.filterByColor = false;
	_params.filterByInertia = false;
	_params.filterByConvexity = false;
	_params.filterByCircularity = true;
	_params.filterByArea = true;
	_params.minCircularity = 0.5;
	_params.maxCircularity = 1.0;
	_params.minArea = 16.0;
	_params.maxArea = 64.0;
        //getConfig();
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
            unsigned int erode_size = 2;
            cv::Mat filt_img;
            cv::Mat element_erode = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(2*erode_size+1, 2*erode_size+1), cv::Point(erode_size, erode_size));
           
            cv::erode(out_img.image, filt_img, element_erode);
            //Dilate image 
            unsigned int dilate_size = 3;
            cv::Mat element_dilate = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(2*dilate_size+1, 2*dilate_size+1), cv::Point(dilate_size, dilate_size));
            cv::dilate(filt_img, out_img.image, element_dilate);
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
	ros::Subscriber _sub_multibeam_scan;
    	ros::Publisher _pub_pointcloud, _pub_image, _pub_image_filtered;

	// Others
	std::string _name;
    	laser_geometry::LaserProjection _projector;
    	tf::TransformListener _listener;

    	double _max_range, _resolution;
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
