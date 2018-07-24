#ifndef SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_
#define SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_

#include "segmatch/segmenters/incremental_segmenter.hpp"

#include <algorithm>

#include <laser_slam/benchmarker.hpp>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in incremental_segmenter.cpp
extern template class IncrementalSegmenter<MapPoint, EuclideanDistance>;
extern template class IncrementalSegmenter<MapPoint, SmoothnessConstraints>;

//=================================================================================================
//    IncrementalSegmenter public methods implementation
//=================================================================================================

template<typename ClusteredPointT, typename PolicyName>
void IncrementalSegmenter<ClusteredPointT, PolicyName>::segment(
    const PointNormals& normals, const std::vector<bool>& is_point_modified, ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
    SegmentedCloud& segmented_cloud, std::vector<Id>& cluster_ids_to_segment_ids,
    std::vector<std::pair<Id, Id>>& renamed_segments) {
  BENCHMARK_BLOCK("SM.Worker.Segmenter");
  renamed_segments.clear();

  // Build partial cluster sets for the old clusters.
  PartialClusters partial_clusters(cluster_ids_to_segment_ids.size());
  for (size_t i = 0u; i < partial_clusters.size(); i++) {
    partial_clusters[i].partial_clusters_set->partial_clusters_indices.insert(i);
    partial_clusters[i].partial_clusters_set->segment_id = cluster_ids_to_segment_ids[i];
  }

  // Find old clusters and new partial clusters.
  growRegions(normals, is_point_modified, cluster_ids_to_segment_ids, cloud,
              points_neighbors_provider, partial_clusters, renamed_segments);

  // Compute and write cluster indices.
  const size_t num_clusters = assignClusterIndices(partial_clusters);
  writeClusterIndicesToCloud(partial_clusters, cloud);

  // Extract the valid segment and add them to the segmented cloud.
  addSegmentsToSegmentedCloud(cloud, partial_clusters, num_clusters, cluster_ids_to_segment_ids,
                              segmented_cloud);
}

//=================================================================================================
//    IncrementalSegmenter private methods implementation
//=================================================================================================

template<typename ClusteredPointT, typename PolicyName>
inline std::pair<Id, Id> IncrementalSegmenter<ClusteredPointT, PolicyName>::mergeSegmentIds(
    const Id id_1, const Id id_2) const {
  if (id_1 == kInvId || id_2 == kInvId) {
    // Invalidated segments stay invalid.
    return  { kInvId, kInvId };
  } else if (id_1 == kNoId) {
    // In case one cluster doesn't belong to a segment keep the only segment ID (if any). No
    // renaming necessary.
    return { kNoId, id_2 };
  } else if (id_2 == kNoId) {
    // In case one cluster doesn't belong to a segment keep the only segment ID (if any). No
    // renaming necessary.
    return { kNoId, id_1 };
  } else {
    // Otherwise take the minimum segment ID (the one that has been around for longer). The segment
    // with maximum ID is renamed to the minimum ID.
    return { std::max(id_1, id_2), std::min(id_1, id_2) };
  }
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::linkPartialClusters(
    const size_t partial_cluster_1_index, const size_t partial_cluster_2_index,
    PartialClusters& partial_clusters, std::vector<std::pair<Id, Id>>& renamed_segments) const {
  // Get pointers to the partial cluster sets
  PartialClustersSetPtr set_1 = partial_clusters[partial_cluster_1_index].partial_clusters_set;
  PartialClustersSetPtr set_2 = partial_clusters[partial_cluster_2_index].partial_clusters_set;

  // Both partial clusters belong to the same set. Nothing to do.
  if (set_1 == set_2) return;

  // Swap the partial cluster indices if it makes the merge operation faster.
  if (set_1->partial_clusters_indices.size() < set_2->partial_clusters_indices.size()) {
    std::swap(set_1, set_2);
  }

  // Move the linked indices from set_2 to set_1 and determine the segment ID.
  set_1->partial_clusters_indices.insert(set_2->partial_clusters_indices.begin(),
                                         set_2->partial_clusters_indices.end());
  Id old_segment_id;
  std::tie(old_segment_id, set_1->segment_id) = mergeSegmentIds(set_1->segment_id,
                                                                set_2->segment_id);

  // Detect if a segment renaming happened
  if (old_segment_id != kNoId && old_segment_id != kInvId)
    renamed_segments.push_back({ old_segment_id, set_1->segment_id });

  // Update all partial clusters contained in set_2 so that they point to set_1.
  for (const auto partial_cluster_index : set_2->partial_clusters_indices) {
    partial_clusters[partial_cluster_index].partial_clusters_set = set_1;
  }
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::growRegionFromSeed(
    const PointNormals& normals, const ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider, const size_t seed_index,
    std::vector<bool>& processed, PartialClusters& partial_clusters,
    std::vector<std::pair<Id, Id>>& renamed_segments) const {
  // Create a new partial cluster.
  partial_clusters.emplace_back();
  PartialCluster& partial_cluster = partial_clusters.back();
  size_t partial_cluster_id = partial_clusters.size() - 1u;
  partial_cluster.partial_clusters_set = std::make_shared<PartialClustersSet>();
  partial_cluster.partial_clusters_set->partial_clusters_indices.insert(partial_cluster_id);

  // Initialize the seeds queue.
  std::vector<size_t>& region_indices = partial_cluster.point_indices;
  std::vector<size_t> seed_queue;
  size_t current_seed_index = 0u;
  seed_queue.push_back(seed_index);
  region_indices.push_back(seed_index);

  // Search for neighbors until there are no more seeds.
  while (current_seed_index < seed_queue.size()) {
    // Search for points around the seed.
    std::vector<int> neighbors_indices = points_neighbors_provider.getNeighborsOf(
        seed_queue[current_seed_index], search_radius_);

    // Decide on which points should we continue the search and if we have to link partial
    // clusters.
    for (const auto neighbor_index : neighbors_indices) {
      if (neighbor_index != -1 && Policy::canGrowToPoint(
          policy_params_, normals, seed_queue[current_seed_index], neighbor_index)) {
        if (isPointAssignedToCluster(cloud[neighbor_index])) {
          // If the search reaches an existing cluster we link to its partial clusters set.
          if (partial_cluster_id != getClusterId(cloud[neighbor_index])) {
            linkPartialClusters(partial_cluster_id, getClusterId(cloud[neighbor_index]),
                                partial_clusters, renamed_segments);
          }
        } else if (!processed[neighbor_index]) {
          // Determine if the point can be used as seed for the region.
          if (Policy::canPointBeSeed(policy_params_, normals, neighbor_index)) {
            seed_queue.push_back(neighbor_index);
          }
          // Assign the point to the current partial cluster.
          region_indices.push_back(neighbor_index);
          processed[neighbor_index] = true;
        }
      }
    }
    ++current_seed_index;
  }
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::growRegions(
    const PointNormals& normals, const std::vector<bool>& is_point_modified,
    const std::vector<Id>& cluster_ids_to_segment_ids, ClusteredCloud& cloud,
    PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
    PartialClusters& partial_clusters, std::vector<std::pair<Id, Id>>& renamed_segments) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.GrowRegions");

  std::vector<bool> processed(cloud.size(), false);
  std::vector<size_t> new_points_indices;
  new_points_indices.reserve(cloud.size());

  for (size_t i = 0u; i < cloud.size(); ++i) {
    if (isPointAssignedToCluster(cloud[i])) {
      // No need to cluster points that are already assigned.
      partial_clusters[getClusterId(cloud[i])].point_indices.push_back(i);
    } else if (Policy::canPointBeSeed(policy_params_, normals, i)) {
      new_points_indices.emplace_back(i);
    }
  }

  // Prepare the seed indices.
  Policy::prepareSeedIndices(normals, new_points_indices.begin(), new_points_indices.end());

  // Process the new points.
  // TODO: The current implementation ignores any change in the normal/curvature of a point,
  // ignoring cases in which changes in the properties of a point would lead to different
  // clustering decisions. It would be nice to add segmentation policies covering this case.
  for (const auto i : new_points_indices) {
    if (!processed[i]) {
      // Mark the point as processed and grow the cluster starting from it.
      processed[i] = true;
      growRegionFromSeed(normals, cloud, points_neighbors_provider, i, processed, partial_clusters,
                         renamed_segments);
    }
  }
}

template<typename ClusteredPointT, typename PolicyName>
inline size_t IncrementalSegmenter<ClusteredPointT, PolicyName>::assignClusterIndices(
    const PartialClusters& partial_clusters) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.AssignClusterIndices");

  // Assign cluster IDs.
  ClusterId next_cluster_id = 1u;
  for (const auto& partial_cluster : partial_clusters) {
    const PartialClustersSetPtr& partial_clusters_set = partial_cluster.partial_clusters_set;
    if (!partial_cluster.point_indices.empty() &&
        partial_clusters_set->cluster_id == kUnassignedClusterId) {
      // Assign a cluster index only if the set didn't get one yet and the partial cluster
      // contains at least one point.
      partial_clusters_set->cluster_id = next_cluster_id;
      ++next_cluster_id;
    }
  }

  return static_cast<size_t>(next_cluster_id);
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::writeClusterIndicesToCloud(
    const PartialClusters& partial_clusters, ClusteredCloud& cloud) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.WriteClusterIndices");

  // Write cluster IDs in the point cloud.
  for (const auto& partial_cluster : partial_clusters) {
    for (const auto point_id : partial_cluster.point_indices) {
      setClusterId(cloud[point_id], partial_cluster.partial_clusters_set->cluster_id);
    }
  }
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::addSegmentsToSegmentedCloud(
    const ClusteredCloud& cloud, const PartialClusters& partial_clusters,
    const size_t num_clusters, std::vector<Id>& cluster_ids_to_segment_ids,
    SegmentedCloud& segmented_cloud) const {
  BENCHMARK_BLOCK("SM.Worker.Segmenter.AddSegments");
  BENCHMARK_RECORD_VALUE("SM.NumClusters", num_clusters);

  // Initially all clusters don't have a segment ID.
  cluster_ids_to_segment_ids = std::vector<Id>(num_clusters, kUnassignedId);
  if (!cluster_ids_to_segment_ids.empty()) cluster_ids_to_segment_ids[0] = kNoId;

  std::vector<Id> segment_ids_to_keep;

  for (size_t i = 0u; i < partial_clusters.size(); i++) {
    const PartialClustersSetPtr& partial_clusters_set = partial_clusters[i].partial_clusters_set;
    const ClusterId cluster_id = partial_clusters_set->cluster_id;

    // Only process clusters once.
    if (cluster_ids_to_segment_ids[cluster_id] != kUnassignedId) continue;

    const Id old_segment_id = partial_clusters_set->segment_id;
    if (old_segment_id == kInvId) {
      // Skip invalidated segments
      cluster_ids_to_segment_ids[cluster_id] = kInvId;
    } else {
      const size_t points_in_cluster = getClusterSize(partial_clusters, i);
      if (points_in_cluster > max_segment_size_) {
        // Invalidate segments with too many points.
        cluster_ids_to_segment_ids[cluster_id] = kInvId;
      } else if (old_segment_id != kNoId || points_in_cluster >= min_segment_size_) {
        // Create the segment, reusing the previous segment ID if present.
        pcl::PointIndices point_indices;
        point_indices.indices = getClusterIndices(partial_clusters, i);
        cluster_ids_to_segment_ids[cluster_id] = segmented_cloud.addSegment(
            point_indices, cloud, old_segment_id);

        segment_ids_to_keep.push_back(cluster_ids_to_segment_ids[cluster_id]);
        BENCHMARK_RECORD_VALUE("SM.SegmentSize", point_indices.indices.size());
      } else {
        // The cluster doesn't have enough points, don't assign a segment yet.
        cluster_ids_to_segment_ids[cluster_id] = kNoId;
      }
    }
  }

  // Delete the segments that we did not keep.
  segmented_cloud.deleteSegmentsExcept(segment_ids_to_keep);
}

template<typename ClusteredPointT, typename PolicyName>
inline size_t IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterSize(
    const PartialClusters& partial_clusters, const size_t partial_cluster_index) const {
  size_t points_in_cluster = 0u;
  const PartialClustersSetPtr& partial_clusters_set =
      partial_clusters[partial_cluster_index].partial_clusters_set;
  for (const auto linked_partial_cluster_index : partial_clusters_set->partial_clusters_indices) {
    points_in_cluster += partial_clusters[linked_partial_cluster_index].point_indices.size();
  }
  return points_in_cluster;
}

template<typename ClusteredPointT, typename PolicyName>
inline std::vector<int> IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterIndices(
    const PartialClusters& partial_clusters, const size_t partial_cluster_index) const {
  const PartialClustersSetPtr& partial_clusters_set =
      partial_clusters[partial_cluster_index].partial_clusters_set;

  std::vector<int> point_indices;
  point_indices.reserve(getClusterSize(partial_clusters, partial_cluster_index));
  for (const auto linked_partial_cluster_index : partial_clusters_set->partial_clusters_indices) {
    point_indices.insert(point_indices.end(),
                         partial_clusters[linked_partial_cluster_index].point_indices.begin(),
                         partial_clusters[linked_partial_cluster_index].point_indices.end());
  }
  return point_indices;
}

template<typename ClusteredPointT, typename PolicyName>
inline bool IncrementalSegmenter<ClusteredPointT, PolicyName>::isPointAssignedToCluster(
    const ClusteredPointT& point) const noexcept {
  return getClusterId(point) != 0u;
}

template<typename ClusteredPointT, typename PolicyName>
inline typename IncrementalSegmenter<ClusteredPointT, PolicyName>::ClusterId
IncrementalSegmenter<ClusteredPointT, PolicyName>::getClusterId(
    const ClusteredPointT& point) const noexcept{
  return Policy::getPointClusterId(point);
}

template<typename ClusteredPointT, typename PolicyName>
inline void IncrementalSegmenter<ClusteredPointT, PolicyName>::setClusterId(
    ClusteredPointT& point, const ClusterId cluster_id) const noexcept {
  Policy::setPointClusterId(point, cluster_id);
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_INCREMENTAL_SEGMENTER_HPP_
