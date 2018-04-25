#ifndef SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_
#define SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_

#define PCL_NO_PRECOMPILE

#include "segmatch/segmenters/euclidean_segmenter.hpp"

#include <laser_slam/benchmarker.hpp>
#include <pcl/segmentation/extract_clusters.h>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in euclidean_segmenter.cpp
extern template class EuclideanSegmenter<MapPoint>;

//=================================================================================================
//    EuclideanSegmenter public methods implementation
//=================================================================================================

template<typename ClusteredPointT>
EuclideanSegmenter<ClusteredPointT>::EuclideanSegmenter(
    const SegmenterParameters& params)
    : params_(params), min_segment_size_(params.min_cluster_size),
      max_segment_size_(params.max_cluster_size),
      radius_for_growing_(params.radius_for_growing) { }

template<typename ClusteredPointT>
void EuclideanSegmenter<ClusteredPointT>::segment(
    const PointNormals& normals, const std::vector<bool>& is_point_modified, ClusteredCloud& cloud,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider, SegmentedCloud& segmented_cloud,
    std::vector<Id>& cluster_ids_to_segment_ids,
    std::vector<std::pair<Id, Id>>& renamed_segments) {
  BENCHMARK_BLOCK("SM.Worker.Segmenter");

  // Clear segments.
  segmented_cloud.clear();

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::extractEuclideanClusters<ClusteredPointT>(
      cloud, points_neighbors_provider.getPclSearchObject(), radius_for_growing_, cluster_indices,
      min_segment_size_,max_segment_size_);

  for (const auto& point_indices : cluster_indices) {
    segmented_cloud.addSegment(point_indices, cloud);
  }

  LOG(INFO) << "Segmentation complete. Found " << cluster_indices.size()
        << " clusters ."<< std::endl;
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_EUCLIDEAN_SEGMENTER_HPP_
