#ifndef SEGMATCH_IMPL_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_
#define SEGMATCH_IMPL_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_

#include <laser_slam/benchmarker.hpp>

#include "segmatch/segmented_cloud.hpp"
#include "segmatch/segmenters/smoothness_constraints_segmenter.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in smoothness_constraints_segmenter.cpp
extern template class SmoothnessConstraintsSegmenter<MapPoint>;

//=================================================================================================
//    SmoothnessConstraintsSegmenter public methods implementation
//=================================================================================================

template<typename ClusteredPointT>
SmoothnessConstraintsSegmenter<ClusteredPointT>::SmoothnessConstraintsSegmenter(
    const SegmenterParameters& params)
  : params_(params), min_segment_size_(params.min_cluster_size),
    max_segment_size_(params.max_cluster_size),
    angle_threshold_(params.sc_smoothness_threshold_deg / 180.0 * M_PI),
    curvature_threshold_(params.sc_curvature_threshold),
    cosine_threshold_(std::cos(angle_threshold_)),
    radius_for_growing_(params.radius_for_growing) {
}

template<typename ClusteredPointT>
void SmoothnessConstraintsSegmenter<ClusteredPointT>::segment(
    const PointNormals& normals, const std::vector<bool>& is_point_modified, ClusteredCloud& cloud,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider, SegmentedCloud& segmented_cloud,
    std::vector<Id>& cluster_ids_to_segment_ids,
    std::vector<std::pair<Id, Id>>& renamed_segments) {
  BENCHMARK_BLOCK("SM.Worker.Segmenter");

  // Clear segments.
  segmented_cloud.clear();

  BENCHMARK_START("SM.Worker.Segmenter.PreparePointIndices");
  // Remove points with high curvature from the list of seeds.
  std::vector<int> indices;
  indices.reserve(cloud.size());
  for (size_t i = 0u; i < normals.size(); ++i) {
    if (canPointBeSeed(normals[i])) indices.emplace_back(i);
  }

  // Sort points in increasing curvature order.
  std::sort(indices.begin(), indices.end(), [&](const int i, const int j) -> bool {
    return normals[i].curvature < normals[j].curvature;
  });
  BENCHMARK_STOP("SM.Worker.Segmenter.PreparePointIndices");

  BENCHMARK_START("SM.Worker.Segmenter.ExtractClusters");
  std::vector<int> number_of_points_in_clusters;
  std::vector<int> point_cluster_ids(cloud.size(), kUnassignedClusterId);
  int cluster_id = 0;

  // Grow clusters from every seed, skipping points that already belong to a cluster.
  for (int seed_index = 0; seed_index < indices.size(); ++seed_index) {
    if (point_cluster_ids[seed_index] == kUnassignedClusterId) {
      number_of_points_in_clusters.push_back(
          growRegionFromSeed(normals, indices[seed_index], cluster_id, points_neighbors_provider,
                             point_cluster_ids));
      ++cluster_id;
    }
  }
  BENCHMARK_STOP("SM.Worker.Segmenter.ExtractClusters");

  // Store the matches in the segmented cloud.
  storeSegments(cloud, point_cluster_ids, number_of_points_in_clusters, segmented_cloud);
}

//=================================================================================================
//    SmoothnessConstraintsSegmenter private methods implementation
//=================================================================================================

template<typename ClusteredPointT>
int SmoothnessConstraintsSegmenter<ClusteredPointT>::growRegionFromSeed(
    const PointNormals& normals, const int seed, const int cluster_id,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider,
    std::vector<int>& point_cluster_ids) const {
  // Initialize the seed queue of the region.
  std::queue<int> region_seeds;
  region_seeds.push(seed);
  point_cluster_ids[seed] = cluster_id;
  int num_points_in_segment = 1;

  // Process all seeds in the queue.
  while (!region_seeds.empty()) {
    int current_seed = region_seeds.front();
    region_seeds.pop();

    // For each neighbor, decide if it belong to the current cluster and if it can be used as seed.
    for (const auto neighbor_index : points_neighbors_provider.getNeighborsOf(
        current_seed, radius_for_growing_)) {
      if (point_cluster_ids[neighbor_index] == kUnassignedClusterId &&
          canGrowToPoint(normals, current_seed, neighbor_index)) {
        point_cluster_ids[neighbor_index] = cluster_id;
        num_points_in_segment++;

        if (canPointBeSeed(normals[neighbor_index]))
          region_seeds.push(neighbor_index);
      }
    }
  }

  return num_points_in_segment;
}

template<typename ClusteredPointT>
inline bool SmoothnessConstraintsSegmenter<ClusteredPointT>::canGrowToPoint(
    const PointNormals& normals, int seed_index, int neighbor_index) const {
  pcl::Vector3fMapConst seed_normal = normals[seed_index].getNormalVector3fMap();
  pcl::Vector3fMapConst neighbor_normal = normals[neighbor_index].getNormalVector3fMap();

  // For computational efficiency, use a threshold on the dot product instead of the angle.
  const float dot_product = std::abs(neighbor_normal.dot(seed_normal));
  return dot_product >= cosine_threshold_;
}

template<typename ClusteredPointT>
inline bool SmoothnessConstraintsSegmenter<ClusteredPointT>::canPointBeSeed(
    const PclNormal& point_normal) const {
  return point_normal.curvature <= curvature_threshold_;
}

template<typename ClusteredPointT>
void SmoothnessConstraintsSegmenter<ClusteredPointT>::storeSegments(
    const ClusteredCloud& cloud, const std::vector<int>& point_cluster_ids,
    const std::vector<int>& number_of_points_in_clusters, SegmentedCloud& segmented_cloud) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.StoreSegments");

  const size_t number_of_clusters = number_of_points_in_clusters.size();
  std::vector<pcl::PointIndices> clusters(number_of_clusters);

  // Reserve space for the clusters.
  for (size_t cluster_index = 0u; cluster_index < number_of_clusters; ++cluster_index) {
    clusters[cluster_index].indices.reserve(number_of_points_in_clusters[cluster_index]);
  }

  // Collect cluster indices.
  for (int point_index = 0; point_index < cloud.size(); ++point_index) {
    const int cluster_id = point_cluster_ids[point_index];
    if (cluster_id != kUnassignedClusterId)
      clusters[cluster_id].indices.emplace_back(point_index);
  }

  // Store the cluster that satisfy the size constraints as segments.
  for (const auto& cluster : clusters) {
    if (cluster.indices.size() >= static_cast<size_t>(min_segment_size_) &&
        cluster.indices.size() <= static_cast<size_t>(max_segment_size_)) {
      segmented_cloud.addSegment(cluster, cloud);
    }
  }
}

template<typename ClusteredPointT>
constexpr int SmoothnessConstraintsSegmenter<ClusteredPointT>::kUnassignedClusterId;

} // namespace segmatch

#endif // SEGMATCH_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_
