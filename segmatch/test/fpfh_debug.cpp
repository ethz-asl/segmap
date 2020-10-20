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

#include <algorithm>

#include "segmatch/segmented_cloud.hpp"
#include "segmatch/features.hpp"

int main(int argc, char **argv) {
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
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

	std::cout<<"Read one pc scan "<<temp_cloud->size()<<std::endl;

	bag.close();

	pcl::io::savePCDFileASCII ("/home/nikhilesh/Documents/segments/test_pcd.pcd", *temp_cloud);

	// Count number of distinct segments (different intensity values).
	std::vector<int> segment_ids;
	for(auto it=temp_cloud->begin(); it!=temp_cloud->end();it++)
	{
		int intensity = int(it->intensity);
		bool already_assigned = (std::find(segment_ids.begin(), segment_ids.end(), intensity) != segment_ids.end());
		if(!already_assigned)
		{
			segment_ids.push_back(intensity);
		}
	}

	std::cout<<"There are "<<segment_ids.size()<<" segments in the point cloud!"<<std::endl;
	
	// Create one point cloud for each segment.
	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>, Eigen::aligned_allocator<pcl::PointXYZRGBA>> cloud_segments(segment_ids.size());
	std::cout<<"Created "<<cloud_segments.size()<<" segments."<<std::endl;

	for(auto pt_it = temp_cloud->begin();pt_it!=temp_cloud->end();pt_it++)
	{
		int segment_id = int(pt_it->intensity);
		auto it = find(segment_ids.begin(), segment_ids.end(), segment_id);
		if (it != segment_ids.end())
		{
			int idx = distance(segment_ids.begin(), it);
			pcl::PointXYZRGBA point;
			point.x = pt_it->x;
			point.y = pt_it->y;
			point.z = pt_it->z;
			cloud_segments[idx].push_back(point);
		}
	}

	for(int i = 0; i<cloud_segments.size();i++)
	{
		if(cloud_segments[i].size()>0)
		{
			std::cout<<"Cloud No. "<<i<<" has "<<cloud_segments[i].size()<<" points."<<std::endl;
			pcl::io::savePCDFileASCII ("/home/nikhilesh/Documents/segments/segment" + std::to_string(i)+".pcd", cloud_segments[i]);
		}
	
	}

	// Pass each point cloud to FPFH object.
	segmatch::DescriptorsParameters dummy_params;
	segmatch::FpfhDescriptor fpfh_tester(dummy_params);

	for(int i=0;i<cloud_segments.size();i++)
	{
		// Convert point cloud to segment.
		segmatch::SegmentView seg_view;
		seg_view.point_cloud = (cloud_segments[i]);
		seg_view.calculateCentroid();
		segmatch::Segment seg;
		seg.clear();
		seg.views.push_back(seg_view);

		// Descriptor.
		segmatch::Features fpfh_feature;

		// Run FPFH.
		fpfh_tester.describe(seg, &fpfh_feature);
	}

	// Retrieve feature.

	// Histogram plot from feature.

	// Save pc to pcd file.

	// Save histogram to file.

	// Close bag file.

	std::cout<<"Hello WELT"<<std::endl;

}