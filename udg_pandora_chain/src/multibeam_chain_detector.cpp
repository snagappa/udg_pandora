#include "ros/ros.h"
#include <ros/console.h>
#include <rosbag/bag.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#include "cola2_lib/cola2_util.h"
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>
#include <pcl_ros/io/pcd_io.h>
#include <iostream>
#include <stdlib.h>

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

		// Subscribe to multibeam laser scan
		_sub_multibeam_scan = _n.subscribe( "/multibeam_scan", 1, &MultibeamChainDetector::updateLaserScan, this);

        _buffer_size = 0;
        _max_range = 4.0;

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
        _projector.transformLaserScanToPointCloud("/world",*scan, cloud, _listener, _max_range);

        _buffer_size++;

        pcl::concatenatePointCloud( _accumulated_point_cloud, cloud, _accumulated_point_cloud);   

        if (_buffer_size > 10) {

            _buffer_size = 0;
            
            _pub_pointcloud.publish(_accumulated_point_cloud);

        }


	}

    void
    filter_chain(){

        
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
    ros::Publisher _pub_pointcloud;

	// Others
	std::string _name;
    laser_geometry::LaserProjection _projector;
    tf::TransformListener _listener;

    double _max_range;
    unsigned int _buffer_size;
    sensor_msgs::PointCloud2 _accumulated_point_cloud;

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
