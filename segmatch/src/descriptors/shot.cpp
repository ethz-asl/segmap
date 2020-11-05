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

  // Get rid off NaNs (Shot doesn't filter them and will break).
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
  cloud->push_back(centroid1);
  cloud_normals->push_back(centroid_normal);

  // Create ShotE class and pass data+normals to it.
  pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
  shot.setInputCloud(cloud);
  shot.setInputNormals(cloud_normals);
  shot.computePointSHOT();  

  //###############################################

  // Return descriptor.
  Eigen::VectorXf shot_vec(30);

  Feature shot_feature("shot");
  for (size_t j = 0u; j < shot_vec.size(); ++j){

      shot_feature.push_back(
      FeatureValue("shot_" + std::to_string(j), double(shot_vec[j])));
      // std::cout<<double(shot_vec[j])<<std::endl;
  }

  features->replaceByName(shot_feature);

  double secondsPassed =  (clock() - startTime) / CLOCKS_PER_SEC;

}

} // namespace segmatch
