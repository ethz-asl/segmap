#include "segmatch/descriptors/fpfh.hpp"
#include "segmatch/parameters.hpp"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/io/pcd_io.h>

int main(int argc, char **argv) {
	// Create Segmatched FPFH object.
	segmatch::DescriptorsParameters dummy_params;
	segmatch::FpfhDescriptor fpfh_tester(dummy_params);

	// Load point clouds from bag file.
	rosbag::Bag bag;
	bag.open("/home/nikhilesh/Documents/segments/segments.bag", rosbag::bagmode::Read);
	std::vector<std::string> topics;
	topics.push_back(std::string("segmatch/source_representation"));
	rosbag::View view(bag, rosbag::TopicQuery(topics));
	sensor_msgs::PointCloud2::ConstPtr input;
	int l=0;
	for(rosbag::MessageInstance const m: rosbag::View(bag))
	{	
		l++;
		std::cout<<"HalloI"<<std::endl;
		input = m.instantiate<sensor_msgs::PointCloud2>();
		if (input!=NULL) 
			{
				std::cout<<"Break"<<std::endl;
				break;
			}
	}

	if(input==NULL)
	{
		std::cout<<"OOOOO"<<std::endl;
	}
	
	pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

	std::cout<<"Read one pc scan "<<temp_cloud->size()<<std::endl;

	bag.close();

	pcl::io::savePCDFileASCII ("/home/nikhilesh/Documents/segments/test_pcd.pcd", *temp_cloud);

	// Count number of distinct segments (different intensity values).

	// Create one point cloud for each segment.

	// Pass each point cloud to FPFH object.

	// Retrieve feature.

	// Histogram plot from feature.

	// Save pc to pcd file.

	// Save histogram to file.

	// Close bag file.

	std::cout<<"Hello WELT"<<std::endl;

}