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
	bool last_ready_;
	pcl::visualization::CloudViewer *viewer_;
	
	struct Feature{
		int feature_idx;
		cv::Point3f pose;
	};
	
	cv::Mat last_descriptors_left_;
	std::vector< cv::Point3f > last_points_3d_;
	std::vector< int > last_points_3d_status_;
		
public:
	  ImageConverter()
		: it_(nh_),
		  img_right_ready_(false),
		  img_left_ready_(false),
		  last_ready_(false)
	  {
		image_sub_left_ = it_.subscribe("/stereo_camera/left/image_rect_color", 1, &ImageConverter::imageCbLeft, this);
		image_sub_right_ = it_.subscribe("/stereo_camera/right/image_rect_color", 1, &ImageConverter::imageCbRight, this);
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
		
		 // Check stereo matches
		 std::vector< cv::DMatch > matches;
		 matcher(descriptors_left_, descriptors_right_, matches);
		 std::cout << "matches size before triangulation: " << matches.size() << std::endl;
		 
		 // Compute 3D points
		 if(matches.size() > 10) {
			 cv::Mat_<float> points = cv::Mat_<float>(3, matches.size());
			 triangulation(matches, points);
			 
			 // Create point cloud and visualize
			 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		
			 for(size_t i = 0; i < matches.size(); i++){
				 if(points[2][i] > 0.75 &&  points[2][i] < 1.5){
					 cloud->points.push_back(pcl::PointXYZ(points[0][i], points[1][i], points[2][i]));		 
				 }
			 }
			 if(cloud->points.size() > 10) viewer_->showCloud(cloud);
			
			 // Create structure to relate descriptors and 3D points
			 std::vector< cv::Point3f > points_3d;
			 std::vector< int > points_3d_status;
			 points_3d.resize(keypoints_left_.size());
			 points_3d_status.resize(keypoints_left_.size());
					 
			 for(size_t i = 0; i < matches.size(); i++){
				 if(points[2][i] > 0.75 &&  points[2][i] < 4.5){
					 // If points_3d_status.at(i) == 1, then feature i has the 3D point points_3d.at(i)
					 points_3d.at(matches[i].queryIdx) = cv::Point3f(points[0][i], points[1][i], points[2][i]);
					 points_3d_status.at(matches[i].queryIdx) = 1;
				 }
			 }
					
			 // Compute Transformation wrt last image
			 if (last_ready_){
				 
				 // Check matches with last image
				 std::vector< cv::DMatch > matches2;
				 matcher(last_descriptors_left_, descriptors_left_, matches2);
				 
				 // Prepare two sets of 3D points
				 std::vector< cv::Point3f > points_1;
				 std::vector< cv::Point3f > points_2;
				 for(size_t i = 0; i < matches2.size(); i++){
					 if((last_points_3d_status_.at(matches2[i].queryIdx) == 1) && (points_3d_status.at(matches2[i].trainIdx) == 1)){
						 points_1.push_back(last_points_3d_.at(matches2[i].queryIdx));
						 points_2.push_back(points_3d.at(matches2[i].trainIdx));
					 }
				 }			 
				 
				 // Compute 3D Affine transformation
				 if(points_1.size() > 10) {
					 std::cout << points_1.size() << " <--> " << points_2.size() << std::endl;
					 std::cout << points_1.at(9) << " -- " << points_2.at(9) << std::endl;
								 
					 cv::Mat aff(3,4,CV_64F);
					 if(computeAffine3dTransformation(points_1, points_2, aff))
					 {
						 std::cout << aff << std::endl;
					 }
				 }
			 }
			 
			 // Save current values as previous
			 descriptors_left_.copyTo(last_descriptors_left_);
			 last_points_3d_.resize(points_3d.size());
			 std::copy(points_3d.begin(), points_3d.end(), last_points_3d_.begin());
			 last_points_3d_status_.resize(points_3d_status.size());
			 std::copy(points_3d_status.begin(), points_3d_status.end(), last_points_3d_status_.begin());
			 last_ready_=true;
		 }
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
 	 
 	int computeAffine3dTransformation(const std::vector<cv::Point3f>& first, const std::vector<cv::Point3f>& second, cv::Mat& aff){
 		std::vector<uchar> inliers;
 		
//		cv::Mat aff(3,4,CV_64F);
// 		std::vector<cv::Point3f> first, second;	
// 		first.push_back(cv::Point3f(0,0,0));
// 		first.push_back(cv::Point3f(1,0,0));
// 		first.push_back(cv::Point3f(0,1,0));
// 		first.push_back(cv::Point3f(0,0,1));
// 		second.push_back(cv::Point3f(1,0,0));
// 		second.push_back(cv::Point3f(2,0,0));
// 		second.push_back(cv::Point3f(1,1,0));
// 		second.push_back(cv::Point3f(1,0,1));
 		 		
 		return cv::estimateAffine3D(first, second, aff, inliers);
 		 		
 	}
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
