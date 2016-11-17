#include "segmatch/segmenters/euclidean_segmenter.hpp"

#include <glog/logging.h>
#include <laser_slam/common.hpp>

namespace segmatch {

EuclideanSegmenter::EuclideanSegmenter()  {}

EuclideanSegmenter::EuclideanSegmenter(const SegmenterParameters& params) :
        kd_tree_(new pcl::search::KdTree<PointI>), params_(params) {
  euclidean_cluster_extractor_.setClusterTolerance(params.ec_tolerance);
  euclidean_cluster_extractor_.setMinClusterSize(params.ec_min_cluster_size);
  euclidean_cluster_extractor_.setMaxClusterSize(params.ec_max_cluster_size);
  euclidean_cluster_extractor_.setSearchMethod(kd_tree_);
}

EuclideanSegmenter::~EuclideanSegmenter() {
  kd_tree_.reset();
}

void EuclideanSegmenter::segment(const PointICloud& cloud,
                                 SegmentedCloud* segmented_cloud) {
  // Clear segments.
  CHECK_NOTNULL(segmented_cloud)->clear();

  laser_slam::Clock clock;
  LOG(INFO) << "Starting euclidean segmentation.";

  PointICloudPtr cloud_ptr(new PointICloud);
  *cloud_ptr = cloud;

  std::vector<pcl::PointIndices> cluster_indices;
  euclidean_cluster_extractor_.setInputCloud(cloud_ptr);
  euclidean_cluster_extractor_.extract(cluster_indices);

  pcl::PointCloud<pcl::PointNormal> cloud_with_normals;
  pcl::copyPointCloud<PointI, pcl::PointNormal>(cloud, cloud_with_normals);
  segmented_cloud->addValidSegments(cluster_indices, cloud_with_normals);

  clock.takeTime();
  LOG(INFO) << "Segmentation complete. Took " << clock.getRealTime() << "ms and found " <<
      cluster_indices.size() << " clusters."<< std::endl;
}

} // namespace segmatch
