#include "segmatch/descriptors/shot.hpp"

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/common.h>
#include <pcl/features/shot.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <ctime>
#include <cmath>

#pragma STDC FENV_ACCESS on

namespace segmatch {

/// \brief Utility function for swapping two values.
template<typename T>
bool swap_if_gt(T& a, T& b) {
  if (a > b) {
    std::swap(a, b);
    return true;
  }
  return false;
}

// ShotDescriptor methods definition
ShotDescriptor::ShotDescriptor(const DescriptorsParameters& parameters) {
  ne_radius_ = parameters.fast_point_feature_histograms_normals_search_radius;
}

void ShotDescriptor::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  std::feclearexcept(FE_ALL_EXCEPT);

  // Do Stuff in here.
  std::cout<<"SHOT START "<<std::endl;
  clock_t startTime = clock(); //Start timer

  // Extract point cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(segment.getLastView().point_cloud, *cloud);

  // Extract surface normals for point cloud.
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_ne(new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod(tree_ne);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  std::cout<<"NE Radius "<<ne_radius_<<std::endl;
  ne.setRadiusSearch(ne_radius_);
  ne.compute(*cloud_normals);

  // Get rid off NaNs.
  pcl::PointCloud<pcl::PointXYZ>::Ptr test_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr test_cloud_normals (new pcl::PointCloud<pcl::Normal>);
  std::vector<int> indices_good_cloud;
  std::vector<int> indices_good_cloud_normals;
  pcl::removeNaNFromPointCloud(*cloud, *test_cloud, indices_good_cloud); 
  pcl::removeNaNNormalsFromPointCloud (*cloud_normals, *test_cloud_normals, indices_good_cloud_normals); 

  int cc1=0;
  int cc2=0;
  for(auto it=cloud->begin();it!=cloud->end();it++)
  {
    if(!pcl_isfinite(it->x) || !pcl_isfinite(it->y) || !pcl_isfinite(it->z))
    {
      cc1++;
    }
  }
  
  for(auto itt=cloud_normals->begin();itt!=cloud_normals->end();itt++)
  {
    if(!pcl_isfinite(itt->normal_x) || !pcl_isfinite(itt->normal_y) || !pcl_isfinite(itt->normal_z))
    {
      cc2++;
    }
  }

  std::cout<<"NanPC: "<<cc1<<" NanPCN: "<<cc2<<std::endl;

  // ToDo(alaturn) Actually filter out NaNs.

  // Get centroid and fake normal as single keypoint.
  PclPoint centroid = segment.getLastView().centroid;
  pcl::Normal centroid_normal(0.0,0.0,1.0); // Fake normal. Shouldn't matter too much for 2D motion..
  pcl::PointXYZ centroid1(centroid.x, centroid.y, centroid.z);

  // A fake point cloud only containing the centroid (= single keypoint).
  pcl::PointCloud<pcl::PointXYZ>::Ptr centroid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  centroid_cloud->push_back(centroid1);
  pcl::PointCloud<pcl::Normal>::Ptr centroid_cloud_normals(new pcl::PointCloud<pcl::Normal>);
  centroid_cloud_normals->push_back(centroid_normal);

  std::cout<<"Size cloud: "<<centroid_cloud->size()<<" size cloud normals: "<<centroid_cloud_normals->size()<<std::endl;
  std::cout<<"Surface size: "<<cloud->size()<<std::endl;  
  // Create ShotE class and pass data+normals to it.
  pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_shot(new pcl::search::KdTree<pcl::PointXYZ>);
  shot.setSearchMethod(tree_shot);
  shot.setInputCloud(centroid_cloud);
  shot.setInputNormals(cloud_normals);  // Is this correct? using different input cloud...
  shot.setSearchSurface(cloud);
  shot.setRadiusSearch(25.0); // ToDo(alaturn) Find actual radius of enclosing sphere!
  pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());
  shot.compute(*descriptors);

  std::cout<<"----------------HALLLOOOOOO--------------"<<std::endl;
  std::cout<<"XYZ: "<<centroid1<<std::endl;
  std::cout<<"Number of histograms "<<descriptors->size()<<std::endl; 
  std::cout<<"HIST1: "<<descriptors->points[0]<<std::endl;
  pcl::SHOT352 descriptor = descriptors->points[0];
  std::cout<<"LOOL: "<<descriptor.getNumberOfDimensions()<<std::endl;
  // std::cout<<" LRF Radius: "<<shot.getLRFRadius()<<std::endl;
  std::cout<<"HIST2: "<<descriptor<<std::endl;
  std::cout<<"HIST3: "<<descriptor.descriptor<<std::endl;
  // Return descriptor.
  Feature shot_feature("shot");
  std::cout<<"YOOOOOOO"<<std::endl;
  for (size_t j = 0u; j < 352; ++j){
      // std::cout<<j<<std::endl;
      shot_feature.push_back(
      FeatureValue("shot_" + std::to_string(j), double(descriptor.descriptor[j])));
      std::cout<<double(descriptor.descriptor[j])<<std::endl;
  }
  std::cout<<"DOOONE"<<std::endl;
  features->replaceByName(shot_feature);

  double secondsPassed =  (clock() - startTime) / CLOCKS_PER_SEC;

}

} // namespace segmatch
