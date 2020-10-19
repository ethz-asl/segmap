#include "segmatch/descriptors/fpfh.hpp"

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/common.h>
#include <pcl/features/fpfh.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
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

// FpfhDescriptor methods definition
FpfhDescriptor::FpfhDescriptor(const DescriptorsParameters& parameters) {}

void FpfhDescriptor::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  std::feclearexcept(FE_ALL_EXCEPT);

  // Do Stuff in here.
  std::cout<<"Start "<<std::endl;
  clock_t startTime = clock(); //Start timer

  // Extract point cloud.
  // PointCloudPtr cloud(new PointCloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(segment.getLastView().point_cloud, *cloud);

  // Extract surface normals for point cloud.
  // ToDo(alaturn) How to handle NaN normals? How to choose radius?  
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_ne(new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod(tree_ne);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.03);
  ne.compute(*cloud_normals);

  // Get centroid of segment.
  PclPoint centroid = segment.getLastView().centroid;
  
  // Get Z-Axis (= fake normal for centroid makes descriptor invariant to centroid normal)
  pcl::Normal centroid_normal(0.0,0.0,1.0);
  pcl::PointXYZ centroid1(centroid.x, centroid.y, centroid.z);
  
  cloud->push_back(centroid1);

  cloud_normals->push_back(centroid_normal);

  // Create FPFHE class and pass data+normals to it.
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(cloud);
  fpfh.setInputNormals(cloud_normals);

  // Create empty kdtree.
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_fpfh(new pcl::search::KdTree<pcl::PointXYZ>);
  fpfh.setSearchMethod(tree_fpfh);

  // Create output dataset.
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

  // Compute largest distance centroid-pt.
  Eigen::Vector4f max_pt_eig;
  pcl::getMaxDistance(*cloud, centroid1.getVector4fMap(), max_pt_eig);
  float max_distance = (Eigen::Vector3f(centroid1.x, centroid1.y, centroid1.z) - Eigen::Vector3f(max_pt_eig[0], max_pt_eig[1], max_pt_eig[2])).norm();

  // Set radius-search to allow for all points.
  fpfh.setRadiusSearch(1.1*max_distance);

  // Only compute SPFH for centroid.
  std::vector<int> indices(cloud->size()-1);  // We don't want to include the last point, which is the centroid.
  std::iota(std::begin(indices), std::end(indices), 0);

  std::cout<<"Test 5!"<<std::endl;

  int nr_subdiv = 11; // ToDo(alaturn) Make param.
  Eigen::MatrixXf hist_f1(1, nr_subdiv), hist_f2(1, nr_subdiv), hist_f3(1, nr_subdiv); 
  Eigen::MatrixXf hist_tot(1, 3*nr_subdiv);
  // Check that last entry = centroid.
  fpfh.computePointSPFHSignature(*cloud, *cloud_normals, int(cloud->size()-1), 0, indices, hist_f1, hist_f2, hist_f3);
  hist_tot << hist_f1, hist_f2, hist_f3;
  Eigen::VectorXf lol(3*nr_subdiv);
  lol = hist_tot.row(0);

  std::cout<<"Test 200!"<<std::endl;

  // // Return descriptor.
  //################ BUG ISOLATION ##############################
  Eigen::VectorXf fpfh_vec(3*nr_subdiv);
  std::cout<<"size(fpfh_vec): "<<fpfh_vec.size()<<" size(hist_f1): "<<hist_f1.size()<<" size(hist_f2): "<<hist_f2.size()<<" size(hist_f3): "<<hist_f3.size()<<std::endl;
  fpfh_vec = hist_tot.row(0);
  // fpfh_vec << hist_f1.row(0), hist_f2.row(0), hist_f3.row(0);
  std::cout<<"Size feature vec: "<<fpfh_vec.size()<<std::endl;
  //#############################################################

  Feature fpfh_feature("fpfh");

  // Sandbox
  std::vector<double> test_fpfh(33, 12); 
  std::generate(test_fpfh.begin(), test_fpfh.end(), std::rand);
  for (size_t j = 0u; j < test_fpfh.size(); ++j){//fpfh_vec.size(); ++j) {

      fpfh_feature.push_back(
      FeatureValue("fpfh_" + std::to_string(j), test_fpfh[j])); // fpfh_vec[j]));
  }

  std::cout<<"Size feature: "<<test_fpfh.size()<<" "<<fpfh_feature.size()<<std::endl;

  features->replaceByName(fpfh_feature);

  double secondsPassed =  (clock() - startTime) / CLOCKS_PER_SEC;

}

} // namespace segmatch
