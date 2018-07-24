#ifndef SEGMATCH_SEGMENTER_HPP_
#define SEGMATCH_SEGMENTER_HPP_

#include <vector>

#include "segmatch/common.hpp"
#include "segmatch/points_neighbors_providers/points_neighbors_provider.hpp"

namespace segmatch {

// Forward declaration to speed up compilation time.
class SegmentedCloud;

/// \brief Interface for point cloud segmenters in SegMatch..
template<typename ClusteredPointT>
class Segmenter {
 public:
  typedef pcl::PointCloud<ClusteredPointT> ClusteredCloud;

  /// \brief Finalizes an instance of the Segmenter class.
  virtual ~Segmenter() = default;

  /// \brief Cluster the given point cloud, writing the found segments in the segmented cloud.
  /// If cluster IDs change, the \c cluster_ids_to_segment_ids mapping is updated accordingly.
  /// \param normals The normal vectors of the point cloud. This can be an empty cloud if the
  /// the segmenter doesn't require normals.
  /// \param is_point_modified Indicates for each point if it has been modified such that its
  /// cluster assignment may change.
  /// \param cloud The point cloud that must be segmented.
  /// \param points_neighbors_provider Object providing nearest neighbors information.
  /// \param segmented_cloud Cloud to which the valid segments will be added.
  /// \param cluster_ids_to_segment_ids Mapping between cluster IDs and segment IDs. Cluster
  /// \c i generates segment \c cluster_ids_to_segments_ids[i]. If
  /// \c cluster_ids_to_segments_ids[i] is equal to zero, then the cluster does not contain enough
  /// points to be considered a segment.
  /// \param renamed_segments Vectors containing segments that got a new ID, e.g. after merging
  /// two or more segments. The ordering of the vector represents the sequence of renaming
  /// operations. The first ID in each pair is the renamed segments, the second ID is the new
  /// segment ID.
  virtual void segment(const PointNormals& normals, const std::vector<bool>& is_point_modified,
                       ClusteredCloud& cloud,
                       PointsNeighborsProvider<ClusteredPointT>& points_neighbors_provider,
                       SegmentedCloud& segmented_cloud,
                       std::vector<Id>& cluster_ids_to_segment_ids,
                       std::vector<std::pair<Id, Id>>& renamed_segments) = 0;
}; // class Segmenter

} // namespace segmatch

#endif // SEGMATCH_SEGMENTER_HPP_
