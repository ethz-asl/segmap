#ifndef SEGMATCH_IMPL_LOCAL_MAP_HPP_
#define SEGMATCH_IMPL_LOCAL_MAP_HPP_

#include "segmatch/local_map.hpp"

#include "laser_slam/benchmarker.hpp"

#include "segmatch/common.hpp"
#include "segmatch/dynamic_voxel_grid.hpp"
#include "segmatch/points_neighbors_providers/kdtree_points_neighbors_provider.hpp"
#include "segmatch/points_neighbors_providers/octree_points_neighbors_provider.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in local_map.cpp
extern template class LocalMap<PclPoint, MapPoint>;

//=================================================================================================
//    LocalMap public methods implementation
//=================================================================================================

template<typename InputPointT, typename ClusteredPointT>
LocalMap<InputPointT, ClusteredPointT>::LocalMap(
    const LocalMapParameters& params, std::unique_ptr<NormalEstimator> normal_estimator)
  : voxel_grid_(params.voxel_size_m, params.min_points_per_voxel)
  , radius_squared_m2_(pow(params.radius_m, 2.0))
  , min_vertical_distance_m_(params.min_vertical_distance_m)
  , max_vertical_distance_m_(params.max_vertical_distance_m)
  , normal_estimator_(std::move(normal_estimator)) {

  // Create the points neighbors provider.
  if (params.neighbors_provider_type == "KdTree") {
    points_neighbors_provider_ = std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>>(
        new KdTreePointsNeighborsProvider<ClusteredPointT>());
  } else if (params.neighbors_provider_type == "Octree") {
    points_neighbors_provider_ = std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>>(
        new OctreePointsNeighborsProvider<ClusteredPointT>(params.voxel_size_m));
  } else {
    LOG(ERROR) << "Invalid points neighbors provider type specified: "
        << params.neighbors_provider_type;
    throw std::invalid_argument("Invalid points neighbors provider type specified: " +
                                params.neighbors_provider_type);
  }
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::updatePoseAndAddPoints(
    const std::vector<InputCloud>& new_clouds, const laser_slam::Pose& pose) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap");

  std::vector<bool> is_point_removed = updatePose(pose);
  std::vector<int> created_points_indices = addPointsAndGetCreatedVoxels(new_clouds);
  std::vector<int> points_mapping = buildPointsMapping(is_point_removed, created_points_indices);

  // Update the points neighbors provider.
  BENCHMARK_START("SM.UpdateLocalMap.UpdatePointsNeighborsProvider");
  getPointsNeighborsProvider().update(getFilteredPointsPtr(), {});
  BENCHMARK_STOP("SM.UpdateLocalMap.UpdatePointsNeighborsProvider");

  // If required, update the normals.
  if (normal_estimator_ != nullptr) {
    BENCHMARK_BLOCK("SM.UpdateLocalMap.EstimateNormals");
    is_normal_modified_since_last_update_ = normal_estimator_->updateNormals(
        getFilteredPoints(), points_mapping, created_points_indices, getPointsNeighborsProvider());
  } else {
    is_normal_modified_since_last_update_ = std::vector<bool>(getFilteredPoints().size(), false);
  }
}

template<typename InputPointT, typename ClusteredPointT>
std::vector<bool> LocalMap<InputPointT, ClusteredPointT>::updatePose(const laser_slam::Pose& pose) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.UpdatePose");

  pcl::PointXYZ position;
  position.x = pose.T_w.getPosition()[0];
  position.y = pose.T_w.getPosition()[1];
  position.z = pose.T_w.getPosition()[2];

  // Remove points according to a cylindrical filter predicate.
  std::vector<bool> is_point_removed = voxel_grid_.removeIf([&](const ClusteredPointT& p) {
    float distance_xy_squared = pow(p.x - position.x, 2.0) + pow(p.y - position.y, 2.0);
    bool remove = distance_xy_squared > radius_squared_m2_
        || p.z - position.z < min_vertical_distance_m_
        || p.z - position.z > max_vertical_distance_m_;
    // TODO: Once we start supporting multiple segmenters working on the same cloud, we will need
    // one \c segment_ids_ vector per segmenter.
    if (remove && p.ed_cluster_id != 0u)
      segment_ids_[p.ed_cluster_id] = kInvId;
    if (remove && p.sc_cluster_id != 0u)
      segment_ids_[p.sc_cluster_id] = kInvId;
    return remove;
  });

  return is_point_removed;
}

template<typename InputPointT, typename ClusteredPointT>
std::vector<int> LocalMap<InputPointT, ClusteredPointT>::addPointsAndGetCreatedVoxels(
    const std::vector<InputCloud>& new_clouds) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.AddNewPoints");

  // Reserve space for the new cloud.
  InputCloud merged_cloud;
  size_t points_count = 0u;
  for (const auto& cloud : new_clouds) points_count += cloud.size();
  merged_cloud.reserve(points_count);

  // Accumulate clouds and insert them in the voxel grid.
  for (const auto& cloud : new_clouds) merged_cloud += cloud;
  std::vector<int> created_points_indices = voxel_grid_.insert(merged_cloud);

  // Record local map metrics.
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.InsertedPoints", merged_cloud.size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.CreatedVoxels", created_points_indices.size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.ActiveVoxels", getFilteredPoints().size());
  BENCHMARK_RECORD_VALUE("SM.UpdateLocalMap.InactiveVoxels",
                         voxel_grid_.getInactiveCentroids().size());

  return created_points_indices;
}

template<typename InputPointT, typename ClusteredPointT>
std::vector<int> LocalMap<InputPointT, ClusteredPointT>::buildPointsMapping(
    const std::vector<bool>& is_point_removed, const std::vector<int>& new_points_indices) {
  BENCHMARK_BLOCK("SM.UpdateLocalMap.BuildPointsMapping");

  // Build a mapping from index in the old point cloud to index in the new point cloud.
  size_t new_point_index = 0u;
  size_t next_inserted_point_index = 0u;
  std::vector<int> mapping(is_point_removed.size());

  for (size_t old_point_index = 0u; old_point_index < is_point_removed.size(); ++old_point_index) {
    if (is_point_removed[old_point_index]) {
      // Mark point as removed.
      mapping[old_point_index] = -1;
    } else {
      while (next_inserted_point_index < new_points_indices.size() &&
          new_points_indices[next_inserted_point_index] == new_point_index) {
        // Skip any inserted point, they don't belong to the mapping.
        ++new_point_index;
        ++next_inserted_point_index;
      }
      mapping[old_point_index] = new_point_index++;
    }
  }

  return mapping;
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::transform(
    const kindr::minimal::QuatTransformationTemplate<float>& transformation) {
  BENCHMARK_BLOCK("SM.TransformLocalMap");
  voxel_grid_.transform(transformation);

  if (normal_estimator_ != nullptr) {
    BENCHMARK_BLOCK("SM.TransformLocalMap.TransformNormals");
    normal_estimator_->notifyPointsTransformed(transformation);
  }
}

template<typename InputPointT, typename ClusteredPointT>
void LocalMap<InputPointT, ClusteredPointT>::clear() {
  voxel_grid_.clear();
  if (normal_estimator_ != nullptr)
    normal_estimator_->clear();
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_LOCAL_MAP_HPP_
