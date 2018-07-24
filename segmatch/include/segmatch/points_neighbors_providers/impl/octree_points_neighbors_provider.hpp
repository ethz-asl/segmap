#ifndef SEGMATCH_IMPL_OCTREE_POINTS_NEIGHBORS_PROVIDER_HPP_
#define SEGMATCH_IMPL_OCTREE_POINTS_NEIGHBORS_PROVIDER_HPP_

#include <glog/logging.h>

#include "segmatch/points_neighbors_providers/octree_points_neighbors_provider.hpp"

namespace segmatch {

// Force the compiler to reuse instantiations provided in octree_points_neighbors_provider.cpp
extern template class OctreePointsNeighborsProvider<MapPoint>;

//=================================================================================================
//    OctreePointsNeighborsProvider public methods implementation
//=================================================================================================

template<typename PointT>
void OctreePointsNeighborsProvider<PointT>::update(
    const typename pcl::PointCloud<PointT>::ConstPtr point_cloud,
    const std::vector<int64_t>& points_mapping) {
  // Build k-d tree
  point_cloud_ = point_cloud;
  octree_.setInputCloud(point_cloud_);
}

template<typename PointT>
const PointNeighbors OctreePointsNeighborsProvider<PointT>::getNeighborsOf(
    const size_t point_index, const float search_radius) {
  CHECK(point_cloud_ != nullptr);

  // Get the neighbors from the octree.
  std::vector<int> neighbors_indices;
  std::vector<float> neighbors_distances;
  octree_.radiusSearch((*point_cloud_)[point_index], search_radius, neighbors_indices,
                        neighbors_distances);

  return neighbors_indices;
}

} // namespace segmatch

#endif // SEGMATCH_IMPL_OCTREE_POINTS_NEIGHBORS_PROVIDER_HPP_
