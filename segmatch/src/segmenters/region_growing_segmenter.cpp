#include "segmatch/segmenters/region_growing_segmenter.hpp"

#include <glog/logging.h>
#include <laser_slam/common.hpp>

namespace segmatch {

RegionGrowingSegmenter::RegionGrowingSegmenter()  {}

RegionGrowingSegmenter::RegionGrowingSegmenter(const SegmenterParameters& params) :
        kd_tree_(new pcl::search::KdTree<PointI>), params_(params) {

  if (params.rg_knn_for_normals == 0) {
    CHECK_NE(params.rg_radius_for_normals, 0.0) << "Wrong parameters for normal estimation.";
    normal_estimator_omp_.setRadiusSearch(params.rg_radius_for_normals);
    LOG(INFO) << "Normals estimation based on radius.";
  } else {
    normal_estimator_omp_.setKSearch(params.rg_knn_for_normals);
    LOG(INFO) << "Normals estimation based on knn.";
  }
  normal_estimator_omp_.setSearchMethod(kd_tree_);
  // Ensure that the normals point to the same direction.
  normal_estimator_omp_.setViewPoint(std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max());

  region_growing_estimator_.setMinClusterSize(params.rg_min_cluster_size);
  region_growing_estimator_.setMaxClusterSize(params.rg_max_cluster_size);
  region_growing_estimator_.setSearchMethod(kd_tree_);
  region_growing_estimator_.setNumberOfNeighbours(params.rg_knn_for_growing);
  region_growing_estimator_.setSmoothnessThreshold(
      params.rg_smoothness_threshold_deg / 180.0 * M_PI);
  region_growing_estimator_.setCurvatureThreshold(params.rg_curvature_threshold);

  region_growing_estimator_.setSmoothModeFlag(true);
  region_growing_estimator_.setCurvatureTestFlag(true);
  region_growing_estimator_.setResidualTestFlag(false);
}

RegionGrowingSegmenter::~RegionGrowingSegmenter() {
  kd_tree_.reset();
}

void RegionGrowingSegmenter::segment(const PointICloud& cloud,
                                     SegmentedCloud* segmented_cloud) {
  // Clear segments.
  CHECK_NOTNULL(segmented_cloud)->clear();

  laser_slam::Clock clock;
  LOG(INFO) << "Starting region growing segmentation.";

  PointICloudPtr cloud_ptr(new PointICloud);
  *cloud_ptr = cloud;

  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  normal_estimator_omp_.setInputCloud(cloud_ptr);
  normal_estimator_omp_.compute(*normals);
  LOG(INFO) << "Normals are computed.";

  // Remove points with high curvature.
  pcl::IndicesPtr indices(new std::vector <int>);
  for (size_t i = 0u; i < normals->size(); ++i) {
    if (normals->points[i].curvature < params_.rg_curvature_threshold) {
      indices->push_back(i);
    }
  }
  if (indices->size() == 0u) {
    LOG(INFO) << "No points with curvature < " << params_.rg_curvature_threshold << ".";
    return;
  }
  LOG(INFO) << "Number of indices " << indices->size();
  LOG(INFO) << "Number of normals " << normals->size();

  region_growing_estimator_.setInputCloud(cloud_ptr);
  region_growing_estimator_.setInputNormals(normals);
  region_growing_estimator_.setIndices(indices);
  std::vector <pcl::PointIndices> clusters;
  region_growing_estimator_.extract(clusters);

  pcl::PointCloud<pcl::PointNormal> cloud_with_normals;
  pcl::copyPointCloud<PointI, pcl::PointNormal>(cloud, cloud_with_normals);
  segmented_cloud->addValidSegments(clusters, cloud_with_normals);

  clock.takeTime();
  LOG(INFO) << "Segmentation complete. Took " << clock.getRealTime() << "ms."<< std::endl;
}

} // namespace segmatch
