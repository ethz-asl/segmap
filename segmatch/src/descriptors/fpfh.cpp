#include "segmatch/descriptors/fpfh.hpp"

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/common.h>
#include <pcl/features/fpfh.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <ctime>
#include <cmath>

#include <csignal>

#include <pcl/pcl_macros.h>


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
  ne.setRadiusSearch(0.5);
  ne.compute(*cloud_normals);

  // ToDo(alaturn) Get rid off NaNs (FPFH doesn't filter them and will break).
  pcl::PointCloud<pcl::PointXYZ>::Ptr test_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr test_cloud_normals (new pcl::PointCloud<pcl::Normal>);
  std::vector<int> indices_good_cloud;
  std::vector<int> indices_good_cloud_normals;
  pcl::removeNaNFromPointCloud(*cloud, *test_cloud, indices_good_cloud); 
  pcl::removeNaNNormalsFromPointCloud (*cloud_normals, *test_cloud_normals, indices_good_cloud_normals); 
  std::cout<<"size(cloudIN): "<<cloud->size()<<" size(cloudOut): "<<test_cloud->size()<<" size(idx): "<<indices_good_cloud.size()<<std::endl;
  std::cout<<"size(cloudNIN): "<<cloud_normals->size()<<" size(cloudNOut): "<<test_cloud_normals->size()<<" size(idxN): "<<indices_good_cloud.size()<<std::endl;

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
  // pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

  // Compute largest distance centroid-pt.
  // Eigen::Vector4f max_pt_eig;
  // pcl::getMaxDistance(*cloud, centroid1.getVector4fMap(), max_pt_eig);
  // float max_distance = (Eigen::Vector3f(centroid1.x, centroid1.y, centroid1.z) - Eigen::Vector3f(max_pt_eig[0], max_pt_eig[1], max_pt_eig[2])).norm();

  // Set radius-search to allow for all points.
  // fpfh.setRadiusSearch(1.1*max_distance);

  // Only compute SPFH for centroid.
  std::vector<int> indices(cloud->size()-1);  // We don't want to include the last point, which is the centroid.
  std::iota(std::begin(indices), std::end(indices), 0);

  // if(indices.size()<130)
  // {
  //   for(int ll=0;ll<indices.size();ll++)
  //   {
  //     std::cout<<"LL: "<<indices[ll]<<std::endl;
  //   }
  // }

  int nr_subdiv = 11; // ToDo(alaturn) Make param.
  Eigen::MatrixXf hist_f1(1, nr_subdiv), hist_f2(1, nr_subdiv), hist_f3(1, nr_subdiv); 
  Eigen::MatrixXf hist_tot(1, 3*nr_subdiv);
  hist_f1.setZero(); 
  hist_f2.setZero(); 
  hist_f3.setZero();
  hist_tot.setZero();

  std::cout<<"CX: "<<centroid.x<<" CY: "<<centroid.y<<" CZ: "<<centroid.z<<std::endl;
  std::cout<<"RealX: "<<(cloud->end()-1)->x<<" RealY: "<<(cloud->end()-1)->y<<" RealZ: "<<(cloud->end()-1)->z<<std::endl;


  // Check that last entry = centroid.
  std::cout<<"size(pc): "<<cloud->size()<<" size(pcN): "<<cloud_normals->size()<<std::endl;
  std::cout<<"F1: "<<hist_f1.rows()<<" "<<hist_f1.cols()<<" F2: "<<hist_f2.rows()<<" "<<hist_f2.cols()<<" F3: "<<hist_f3.rows()<<" "<<hist_f3.cols()<<std::endl;
  std::raise(SIGINT);
  fpfh.computePointSPFHSignature(*cloud, *cloud_normals, cloud->size()-1, 0, indices, hist_f1, hist_f2, hist_f3);
  std::cout<<"sum(F1): "<<(hist_f1.row(0)).sum()<<" sum(F2): "<<(hist_f2.row(0)).sum()<<" sum(F3): "<<(hist_f3.row(0)).sum()<<std::endl;
  // for(int i=0;i<nr_subdiv;i++)
  // {
  //   if(
  //     (hist_f1(i)>1000.0 || hist_f1(i)<-1000.0 || (hist_f1(i)<0.001 && hist_f1(i)>-0.001)) ||
  //     (hist_f2(i)>1000.0 || hist_f2(i)<-1000.0 || (hist_f2(i)<0.001 && hist_f2(i)>-0.001)) ||
  //     (hist_f3(i)>1000.0 || hist_f3(i)<-1000.0 || (hist_f3(i)<0.001 && hist_f3(i)>-0.001))
  //     )
  //   {
  //     std::cout<<"OHOHOHOHOH "<<hist_f1(i)<<" "<<hist_f2(i)<<" "<<hist_f3(i)<<std::endl;
  //   }

  // }

  hist_tot << hist_f1, hist_f2, hist_f3;

  // Return descriptor.
  Eigen::VectorXf fpfh_vec(3*nr_subdiv);
  fpfh_vec = hist_tot.row(0);
  std::cout << "Feature = " << hist_f1.row(0)<<hist_f2.row(0)<<hist_f3.row(0)<< std::endl;

  Feature fpfh_feature("fpfh");

  for (size_t j = 0u; j < fpfh_vec.size(); ++j){

      fpfh_feature.push_back(
      FeatureValue("fpfh_" + std::to_string(j), double(fpfh_vec[j])));
  }

  features->replaceByName(fpfh_feature);

  double secondsPassed =  (clock() - startTime) / CLOCKS_PER_SEC;

}

} // namespace segmatch
