// This program reads in a rosbag, which contains a pointcloud consisting of segments (segment-ID = intensity).
// It split the cloud into segments, extracts the FPFH descriptor from each segment and saves the segment.

#include <algorithm>
#include<fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

#include "segmatch/descriptors/fpfh.hpp"
#include "segmatch/features.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

int main(int argc, char **argv) {
	// Load point cloud from bag file.
	rosbag::Bag bag;
	bag.open("/home/nikhilesh/Documents/segments/segments.bag", rosbag::bagmode::Read);
	std::vector<std::string> topics;
	topics.push_back(std::string("segmatch/source_representation"));
	rosbag::View view(bag, rosbag::TopicQuery(topics));
	sensor_msgs::PointCloud2::ConstPtr input;
	// ToDo(alaturn) Right now, only works if there is only a single topic on the bag. Filter out by topic...
	for(rosbag::MessageInstance const m: rosbag::View(bag))
	{	
		input = m.instantiate<sensor_msgs::PointCloud2>();
		if (input!=NULL) 
			{
				std::cout<<"Break"<<std::endl;
				break;
			}
	}

	if(input==NULL)
	{
		std::cout<<"Something went wrong!"<<std::endl;
		return 0;
	}
	
	// Conver to PCL standard.
	pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

	std::cout<<"Number of points: "<<temp_cloud->size()<<std::endl;

	bag.close();

	pcl::io::savePCDFileASCII ("/home/nikhilesh/Documents/segments/fpfh_output/full_pcd.pcd", *temp_cloud);

	// Count number of segments (ID = intensity).
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

	// Save also each segment to file for later visualization.
	for(int i = 0; i<cloud_segments.size();i++)
	{
		if(cloud_segments[i].size()>0)
		{
			// std::cout<<"Cloud No. "<<i<<" has "<<cloud_segments[i].size()<<" points."<<std::endl;
			pcl::io::savePCDFileASCII ("/home/nikhilesh/Documents/segments/fpfh_output/segment" + std::to_string(i)+".pcd", cloud_segments[i]);
		}
	}

	// Extract the FPFH descriptor for each segment.
	segmatch::DescriptorsParameters dummy_params;
	segmatch::FpfhDescriptor fpfh_tester(dummy_params);
	Eigen::MatrixXf fpfh_descriptors(33,cloud_segments.size());
	fpfh_descriptors.setZero();

	for(int i=0;i<cloud_segments.size();i++)
	{
		// Convert point cloud to segmatch-segment.
		segmatch::SegmentView seg_view;
		seg_view.point_cloud = (cloud_segments[i]);
		seg_view.calculateCentroid();
		segmatch::Segment seg;
		seg.clear();
		seg.views.push_back(seg_view);

		// Compute FPFH.
		segmatch::Features fpfh_features;
		fpfh_tester.describe(seg, &fpfh_features);
		// std::cout<<"Computed Features: "<<fpfh_features.size()<<std::endl;

		// Extract descriptor.
		std::vector<segmatch::FeatureValueType> descriptor = fpfh_features.asVectorOfValues();
		// std::cout<<"Size Descriptor: "<<descriptor.size()<<std::endl;

		// // Visualize histogram.
  		// pcl::FPFHSignature33 fpfh_descriptor;
		for (int k=0; k<33; k++)
		{
		   fpfh_descriptors(k,i) = float(descriptor[k]);
		  // fpfh_descriptor.histogram[k] = float(descriptor[k]);
		}
		// pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
		// descriptors->push_back(fpfh_descriptor);
		// std::cout<<"Size Hist "<<descriptors->size()<<std::endl;
		// pcl::visualization::PCLPlotter plotter;
		// plotter.addFeatureHistogram(*descriptors, 33);
		// plotter.plot();
	}

	// std::cout<<fpfh_descriptors<<std::endl;

	// Compute cross distances.
	Eigen::MatrixXf distance_matrix(fpfh_descriptors.cols(),fpfh_descriptors.cols());
	for(int i=0;i<fpfh_descriptors.cols();i++)
	{
		for(int j=0;j<fpfh_descriptors.cols();j++)
			{
				// Compute norm(i,j).
				double uu;
				float distance = (fpfh_descriptors.col(i)-fpfh_descriptors.col(j)).norm();
				distance_matrix(i,j) = distance;
			}
	}

	// std::cout<<distance_matrix<<std::endl;

	// Save cross-distance matrix to CSV file.
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
	std::ofstream file("/home/nikhilesh/Documents/segments/fpfh_output/cross_distances.csv");
    if (file.is_open())
    {
        file << distance_matrix.format(CSVFormat);
        file.close();
    }


	// Retrieve feature.

	// Histogram plot from feature.

	// Save pc to pcd file.

	// Save histogram to file.

	// Close bag file.
}