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

#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/pcl_plotter.h>

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
FpfhDescriptor::FpfhDescriptor(const DescriptorsParameters& parameters) {
  ne_radius_ = parameters.fast_point_feature_histograms_normals_search_radius;
}

void FpfhDescriptor::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  std::feclearexcept(FE_ALL_EXCEPT);

  // Do Stuff in here.
  // std::cout<<"FPFH START "<<std::endl;
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
  // std::cout<<"Raidus "<<ne_radius_<<std::endl;
  ne.setRadiusSearch(ne_radius_);  // ToDo(alaturn) Make adaptive or param.
  ne.compute(*cloud_normals);

  // Get rid off NaNs (FPFH doesn't filter them and will break).
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

  // std::cout<<"NanPC: "<<cc1<<" NanPCN: "<<cc2<<std::endl;

  // ToDo(alaturn) Actually filter out NaNs.

  // Get centroid of segment.
  PclPoint centroid = segment.getLastView().centroid;

  // Viz Sandbox.
  // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // viewer->addPointCloud<pcl::PointXYZ> (cloud,"sample cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  // // viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, cloud_normals, 10, 0.05, "normals");
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();
  // viewer->setCameraPosition(centroid.x, centroid.y, centroid.z-2.0, 0,0,1, 0);

  // while (!viewer->wasStopped ())
  // {
  //   viewer->spinOnce (100);
  //   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  // }

  // raise(SIGINT);
  
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

  // Only compute SPFH for centroid.
  std::vector<int> indices(cloud->size()-1);  // We don't want to include the last point, which is the centroid.
  std::iota(std::begin(indices), std::end(indices), 0);


  int nr_subdiv = 11; // ToDo(alaturn) Make param.
  Eigen::MatrixXf hist_f1(1, nr_subdiv), hist_f2(1, nr_subdiv), hist_f3(1, nr_subdiv); 
  Eigen::MatrixXf hist_tot(1, 3*nr_subdiv);
  hist_f1.setZero(); 
  hist_f2.setZero(); 
  hist_f3.setZero();
  hist_tot.setZero();

  // Check that last entry = centroid.
  // std::cout<<"size(pc): "<<cloud->size()<<" size(pcN): "<<cloud_normals->size()<<std::endl;
  // std::cout<<"F1: "<<hist_f1.rows()<<" "<<hist_f1.cols()<<" F2: "<<hist_f2.rows()<<" "<<hist_f2.cols()<<" F3: "<<hist_f3.rows()<<" "<<hist_f3.cols()<<std::endl;
  // std::raise(SIGINT);
  fpfh.computePointSPFHSignature(*cloud, *cloud_normals, (cloud->size())-1, 0, indices, hist_f1, hist_f2, hist_f3);
  // std::cout<<"sum(F1): "<<(hist_f1.row(0)).sum()<<" sum(F2): "<<(hist_f2.row(0)).sum()<<" sum(F3): "<<(hist_f3.row(0)).sum()<<std::endl;
  
  // ToDo(alatur) Check that each histograms sums up to ~100.

  hist_tot << hist_f1, hist_f2, hist_f3;

  // Return descriptor.
  Eigen::VectorXf fpfh_vec(3*nr_subdiv);
  fpfh_vec = hist_tot.row(0);
  // std::cout<<"Sum(FeatureTot): "<<(hist_tot.row(0)).sum()<<std::endl;
  // std::cout << "FeatureTot = " <<hist_tot.row(0)<<std::endl; //hist_f1.row(0)<<hist_f2.row(0)<<hist_f3.row(0)<< std::endl;

  Feature fpfh_feature("fpfh");
  for (size_t j = 0u; j < fpfh_vec.size(); ++j){

      fpfh_feature.push_back(
      FeatureValue("fpfh_" + std::to_string(j), double(fpfh_vec[j])));
      // std::cout<<double(fpfh_vec[j])<<std::endl;
  }

  features->replaceByName(fpfh_feature);

  // Viz histogram.
  // pcl::FPFHSignature33 testss;
  // for (int ss=0; ss<33; ss++)
  // {
  //   testss.histogram[ss]=fpfh_vec[ss];
  // }
  // pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
  // descriptors->push_back(testss);
  // std::cout<<"Size Hist "<<descriptors->size()<<std::endl;
  // pcl::visualization::PCLPlotter plotter;
  // plotter.addFeatureHistogram(*descriptors, 33);
  // plotter.plot();

  //raise(SIGINT);

  double secondsPassed =  (clock() - startTime) / CLOCKS_PER_SEC;

}

} // namespace segmatch
