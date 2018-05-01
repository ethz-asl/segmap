#ifndef SEGMATCH_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_
#define SEGMATCH_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_

#include <string>

#include "segmatch/parameters.hpp"
#include "segmatch/common.hpp"
#include "segmatch/segmented_cloud.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

// Forward declaration to speed up compilation time.
class SegmentedCloud;

/// \brief Region growing segmenter with smoothness constraints. Implementation of the approach
/// described in "Segmentation of point clouds using smoothness constraint", T. Rabbania,
/// F. A. van den Heuvelb, G. Vosselmanc.
template<typename ClusteredPointT>
class SmoothnessConstraintsSegmenter : public Segmenter<ClusteredPointT> {
 public:
  typedef pcl::PointCloud<ClusteredPointT> ClusteredCloud;

  static_assert(pcl::traits::has_xyz<ClusteredPointT>::value,
                "RegionGrowingSegmenter requires ClusteredPointT to contain XYZ "
                "coordinates.");

  /// \brief Initializes a new instance of the SmoothnessConstraintsSegmenter class.
  /// \param params The parameters of the segmenter.m
  explicit SmoothnessConstraintsSegmenter(const SegmenterParameters& params);

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
  void segment(const PointNormals& normals, const std::vector<bool>& is_point_modified,
               ClusteredCloud& cloud, PointsNeighborsProvider<MapPoint>& points_neighbors_provider,
               SegmentedCloud& segmented_cloud, std::vector<Id>& cluster_ids_to_segment_ids,
               std::vector<std::pair<Id, Id>>& renamed_segments) override;

 private:
  // Identify a cluster starting from the specified seed using the region-growing rules.
  int growRegionFromSeed(const PointNormals& normals, int seed, int cluster_id,
                         PointsNeighborsProvider<MapPoint>& points_neighbors_provider,
                         std::vector<int>& point_cluster_ids) const;

  // Determine if it's allowed to grow from a seed to a point.
  bool canGrowToPoint(const PointNormals& normals, int seed_index, int neighbor_index) const;

  // Determine if a point satisfies the requirements for being used as a seed.
  bool canPointBeSeed(const PclNormal& point_normal) const;

  // Transfer the segments to the segmented cloud.
  void storeSegments(const ClusteredCloud& cloud, const std::vector<int>& point_cluster_ids,
                     const std::vector<int>& number_of_points_in_clusters,
                     SegmentedCloud& segmented_cloud) const;

  // Parameters and shortcuts.
  const SegmenterParameters params_;
  const int min_segment_size_;
  const int max_segment_size_;
  const float angle_threshold_;
  const float curvature_threshold_;
  const float cosine_threshold_;
  const float radius_for_growing_;

  static constexpr int kUnassignedClusterId = -1;
}; // class SmoothnessConstraintsSegmenter

} // namespace segmatch

#endif // SEGMATCH_SMOOTHNESS_CONSTRAINTS_SEGMENTER_HPP_
