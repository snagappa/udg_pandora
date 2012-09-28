#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "cvaux.h"

#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>



namespace enc = sensor_msgs::image_encodings;

static const char WINDOW[] = "Image window";

class ImageConverter
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_left_;
  	image_transport::Subscriber image_sub_right_;
	cv_bridge::CvImagePtr img_left_;
	std::vector<cv::KeyPoint> keypoints_left_;
	cv::Mat descriptors_left_;
	cv_bridge::CvImagePtr img_right_;
	std::vector<cv::KeyPoint> keypoints_right_;
	cv::Mat descriptors_right_;
	bool img_right_ready_;
	bool img_left_ready_;
	pcl::visualization::CloudViewer *viewer_;
	
	struct Feature{
		int feature_idx;
		cv::Point3f pose;
	};
	
	std::vector< Feature > last_features_;
	cv::Mat last_descriptors_left_;
	std::vector<cv::KeyPoint> last_keypoints_left_;
		
public:
	  ImageConverter()
		: it_(nh_),
		  img_right_ready_(false),
		  img_left_ready_(false)
	  {
		image_sub_left_ = it_.subscribe("/stereo_down/left/image_rect_color", 1, &ImageConverter::imageCbLeft, this);
		image_sub_right_ = it_.subscribe("/stereo_down/right/image_rect_color", 1, &ImageConverter::imageCbRight, this);
		cv::namedWindow(WINDOW);
		viewer_ = new pcl::visualization::CloudViewer("Viewer");
	  }
	  

	  ~ImageConverter()
	  {
		cv::destroyWindow(WINDOW);
	  }
	  

	 void imageCbLeft(const sensor_msgs::ImageConstPtr& msg)
	  {		
		try
		{
		  img_left_ = cv_bridge::toCvCopy(msg, enc::BGR8);
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}
		
		featuresAndDescriptors(img_left_, keypoints_left_, descriptors_left_);
		
		// Left camera is ready, ask for the right
		img_left_ready_ = true;
		if(img_right_ready_) compute3dPoints();
	  }

	 
	 void imageCbRight(const sensor_msgs::ImageConstPtr& msg)
	  {
		try
		{
		  img_right_ = cv_bridge::toCvCopy(msg, enc::BGR8);
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}
	
		featuresAndDescriptors(img_right_, keypoints_right_, descriptors_right_);
		
		cv::imshow(WINDOW, img_right_->image);
		cv::waitKey(3);
				
		// Right camera is ready, ask for the left
		img_right_ready_ = true;
		if(img_left_ready_) compute3dPoints();		
	  }

	 
	 void compute3dPoints()
	 {
		 img_left_ready_ = false;
		 img_right_ready_ = false;
		 std::vector< cv::DMatch > matches;
		
		 matcher(descriptors_left_, descriptors_right_, matches);
		 std::cout << "matches size before triangulation: " << matches.size() << std::endl;
		 
		 cv::Mat_<float> points = cv::Mat_<float>(3, matches.size());
		 triangulation(matches, points);
		 
		 //Create point cloud
		 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		 
		 for(size_t i = 0; i < matches.size(); i++){
			 if(points[2][i] > 0.75 &&  points[2][i] < 1.5){
				 cloud->points.push_back(pcl::PointXYZ(points[0][i], points[1][i], points[2][i]));
			 }
		 }
		 std::cout << "Number of points: " << cloud->points.size() << std::endl;
		 if(cloud->points.size() > 20) viewer_->showCloud(cloud);
	 }
	 
	 
	 void triangulation(std::vector< cv::DMatch >& matches, cv::Mat_<float>& points)
	 {
		 // Compute projection matrices 
		 // TODO: Compute them from camera_info msgs
		 cv::Mat_<float> proj_mat1 = (cv::Mat_<float>(3, 4) << 724.2, 0., 520.08, 0., 0., 722.33, 385.77, 0., 0., 0., 1., 0.);
		 cv::Mat_<float> proj_mat2 = (cv::Mat_<float>(3, 4) << 724.2 ,0. ,520.08, -82.25, 0., 722.33, 385.77, 0., 0., 0., 1., 0.);
		         
		 // Compute 2d points 
		 cv::Mat_<float> proj_point1 = cv::Mat_<float>(2, matches.size());
		 cv::Mat_<float> proj_point2 = cv::Mat_<float>(2, matches.size());
		 		 
		 for(size_t i = 0; i < matches.size(); i++){
			 // Check epipolar
			 if(fabs(keypoints_left_[matches[i].queryIdx].pt.y - keypoints_right_[matches[i].trainIdx].pt.y) < 2) {
				 proj_point1[0][i] = keypoints_left_[matches[i].queryIdx].pt.x;
				 proj_point1[1][i] = keypoints_left_[matches[i].queryIdx].pt.y;
				 proj_point2[0][i] = keypoints_right_[matches[i].trainIdx].pt.x;
				 proj_point2[1][i] = keypoints_right_[matches[i].trainIdx].pt.y;			 
			 }
		 }

		 // Compute triangulation
		 cv::Mat_<float> homogeneous = cv::Mat_<float>(4, matches.size());
		 cv::triangulatePoints(proj_mat1, proj_mat2, proj_point1, proj_point2, homogeneous);
		 
		 // TODO: CHECK --> cv::convertPointsFromHomogeneous(homogeneous, points);		
		 for(int i = 0; i < homogeneous.cols; i++) {
			 if (fabs(homogeneous[3][i]) > 0.00001){
				 points[0][i] = homogeneous[0][i]/homogeneous[3][i];
			 	 points[1][i] = homogeneous[1][i]/homogeneous[3][i];
			 	 points[2][i] = homogeneous[2][i]/homogeneous[3][i];
			 }
			 else {
				 points[0][i] = 0.0;
				 points[1][i] = 0.0;
				 points[2][i] = 0.0;
			 }
		 }
	 }
	 
	 
	 void matcher(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, std::vector< cv::DMatch >& matches){
		 //Create a matcher and ask for the 2 best matches of each feature point
		 cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		 std::vector< std::vector< cv::DMatch > > raw_matches;
		 matcher->knnMatch(query_descriptors, train_descriptors, raw_matches, 2);
		 
		  // If the first match is much better than the second keep it, otherwise drop it
		 for(size_t i = 0; i < raw_matches.size(); i++) {
			 if(raw_matches[i][0].distance / raw_matches[i][1].distance < 0.7){
				 matches.push_back(raw_matches[i][0]);
			 }			 
		 }
	 }
	 
	 
 	 void featuresAndDescriptors(cv_bridge::CvImagePtr& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
 	 {
		// Color to gray scale
 		cv::Mat img_gray;
		cv::cvtColor(img->image, img_gray, CV_RGB2GRAY);
		
		// Extract keypoints
		//cv::FAST(img_gray, keypoints, 30);
		
		// Compute keypoints descriptors
		//cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("ORB");
		//descriptorExtractor->compute(img_gray, keypoints, descriptors);
		
		// Using ORB
		cv::ORB orb_extractor;
		orb_extractor(img_gray, cv::Mat(), keypoints, descriptors);
				
		// Draw keypoints in color image
		for(size_t i = 0; i < keypoints.size(); ++i) {
			const cv::KeyPoint& kp = keypoints[i];
		 	cv::circle(img->image, kp.pt, 2, CV_RGB(255, 0, 0));       
		}
 	}
 	 
 	void computeAffine3dTransformation(){
 		std::vector<cv::Point3f> first, second;
 		std::vector<uchar> inliers;
 		cv::Mat aff(3,4,CV_64F);

 		first.push_back(cv::Point3f(0,0,0));
 		first.push_back(cv::Point3f(1,0,0));
 		first.push_back(cv::Point3f(0,1,0));
 		first.push_back(cv::Point3f(0,0,1));
 		second.push_back(cv::Point3f(1,0,0));
 		second.push_back(cv::Point3f(2,0,0));
 		second.push_back(cv::Point3f(1,1,0));
 		second.push_back(cv::Point3f(1,0,1));
 		 		
 		int ret = cv::estimateAffine3D(first, second, aff, inliers);
 		std::cout << "MAT: " << aff << std::endl;
 		std::cout << "ret: " << ret << std::endl;
 		 		
 	}
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
