#ifndef SEGMATCH_SEGMENTED_CLOUD_HPP_
#define SEGMATCH_SEGMENTED_CLOUD_HPP_

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/common.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "segmatch/common.hpp"
#include "segmatch/utilities.hpp"
#include "segmatch/features.hpp"

namespace segmatch {

/// \brief Triggered when the amount of segments becomes too high.
extern bool g_too_many_segments_to_store_ids_in_intensity;

struct SegmentView {

  void calculateCentroid();

  /// Points
  pcl::PointCloud<PclPoint> point_cloud;
  pcl::PointCloud<PclPoint> point_cloud_to_publish;
  pcl::PointCloud<PclPoint> reconstruction;
  pcl::PointCloud<PclPoint> reconstruction_compressed;
  
  Features features;
  PclPoint centroid = PclPoint(0,0,0);
  // Time at which the segment was created.
  laser_slam::Time timestamp_ns;
  // Trajectory pose to which the segment is linked.
  laser_slam::SE3 T_w_linkpose;

  // Number of occupied voxels during voxelization for CNN descriptor.
  unsigned int n_occupied_voxels;

  // Number of points the segment had when last described
  unsigned int n_points_when_last_described = 0u;

  unsigned int semantic = 0u;
};

struct Segment {
  bool empty() const { return views.empty(); }
  void clear() {
    segment_id = kNoId;
    views.clear();
  }
  bool hasValidId() const { return segment_id != kNoId && segment_id != kInvId; }

  const SegmentView& getLastView() const { return views.back(); }

  /// \brief Get a reference to the last view.
  /// \remark This function is safe IFF the vector is not modified at all while somebody is
  /// working with this reference.
  SegmentView& getLastView() { return views.back(); }

  Id segment_id = kNoId;

  // Normals
  pcl::PointCloud<pcl::Normal> normals;
  // Overlap with other segments at each point.
  std::vector<Id> overlap;

  laser_slam::Aligned<std::vector, SegmentView> views;
  unsigned int track_id;
};

class SegmentedCloud {
 public:
  SegmentedCloud(const bool keep_only_last_view = true) :
    keep_only_last_view_(keep_only_last_view) {};
  Id getNextId(const Id& begin_counting_from_this_id = 0);
  SegmentedCloud& operator+= (const SegmentedCloud& rhs) {
    for (auto& id_segment: rhs.valid_segments_) {
      this->addValidSegment(id_segment.second);
    }
    return *this;
  }
  void addSegmentedCloud(const SegmentedCloud& segmented_cloud_to_add,
                         const std::vector<std::pair<Id, Id>>& renamed_segments = { });

  /// \brief Adds or updates a segment in the cloud.
  /// \param segment_to_add Indices of the points belonging to the segment.
  /// \param reference_cloud Point cloud from which the segment is extracted.
  /// \param segment_id Id of the segment. If a valid Id is specified, the
  /// segment will be added as an additional view of the same segment. Default
  /// is \e kNoId.
  /// \returns ID of the added segment.
  template <typename PointCloudT>
  Id addSegment(const pcl::PointIndices& segment_to_add,
                const PointCloudT& reference_cloud,
                const Id segment_id = kNoId);

  /// \brief Adds a copy of a segment to this cloud.
  ///        This causes two segments to exist with the same Id.
  void addValidSegment(const Segment& segment_to_add);
  size_t getNumberOfValidSegments() const;
  bool empty() const { return getNumberOfValidSegments() == 0; }
  bool findValidSegmentById(const Id segment_id, Segment* result) const;
  bool findValidSegmentPtrById(const Id segment_id, Segment** result);
  void deleteSegmentsById(const std::vector<Id>& ids, size_t* n_removals=NULL);

  void deleteSegmentsExcept(const std::vector<Id>& segment_ids_to_keep);

  /// brief Clear the segmented cloud.
  void clear();
  void calculateSegmentCentroids();
  void transform(const Eigen::Matrix4f& transform_matrix) {
    for (auto& id_segment : valid_segments_) {
      for (auto& view : id_segment.second.views) {
        pcl::transformPointCloud(view.point_cloud, view.point_cloud,
                                 transform_matrix);
        pcl::transformPointCloud(view.reconstruction, view.reconstruction,
                                 transform_matrix);
      }
    }
    calculateSegmentCentroids(); // TODO: is transforming the centroids faster?
  }
  SegmentedCloud transformed(const Eigen::Matrix4f& transform_matrix) const {
    SegmentedCloud self_copy = *this;
    self_copy.transform(transform_matrix);
    return self_copy;
  }

  PointCloud centroidsAsPointCloud(
      std::vector<Id>& segment_id_for_each_point) const;

  // Get the centroids which linkpose falls within a radius.
  PointCloud centroidsAsPointCloud(const laser_slam::SE3& center,
                                   double maximum_linkpose_distance,
                                   std::vector<Id>* segment_id_for_each_point_ptr=NULL) const;

  bool findNearestSegmentsToPoint(const PclPoint& point,
                                  unsigned int n_closest_segments,
                                  double maximum_centroid_distance_m,
                                  std::vector<Id>* ids,
                                  std::vector<double>* distances) const;

  // TODO group in one segment parameters struct?
  void setTimeStampOfSegments(const laser_slam::Time& timestamp_ns);

  void setLinkPoseOfSegments(const laser_slam::SE3& link_pose);

  void setTrackId(unsigned int track_id);

  void updateSegments(const std::vector<laser_slam::Trajectory>& trajectories);

  std::unordered_map<Id, Segment>::const_iterator begin() const {
    return valid_segments_.begin();
  }

  std::unordered_map<Id, Segment>::const_iterator end() const {
    return valid_segments_.end();
  }

  std::unordered_map<Id, Segment>::iterator begin() {
    return valid_segments_.begin();
  }

  std::unordered_map<Id, Segment>::iterator end() {
    return valid_segments_.end();
  }

  bool contains(const Id segment_id) const {
    return valid_segments_.find(segment_id) != valid_segments_.end();
  }

  /// \brief Gets the number of segments stored in the segmented cloud.
  /// \returns Number of segments stored in the segmented cloud.
  size_t size() const { return valid_segments_.size(); }

  /// \brief Reset the segments ID counter to the specified ID.
  /// \param new_id Value of the next ID that will be assigned to a segment.
  void resetSegmentIdCounter(const Id new_id = 1) { current_id_ = new_id - 1; }

  /// \brief Gets the number of segment pairs whose distance between centroids is less or equal
  /// \c max_distance.
  /// \param max_distance The maximum distance so that two segments can be considered close.
  /// \returns The number of close segment pairs.
  size_t getCloseSegmentPairsCount(float max_distance) const;

  void eraseSegmentById(Id id) { valid_segments_.erase(id); };
  
  // TODO RD Solve the need for cleaning empty segments.
  void cleanEmptySegments();

 private:
  std::unordered_map<Id, Segment> valid_segments_;
  static Id current_id_;
  bool keep_only_last_view_;

  // Create a new view when the number of points increased by this ratio.
  // Currently only used for exporting the run data.
  static constexpr double min_change_to_add_new_view = 1.10;
  
  static constexpr unsigned int publish_every_x_points = 20;
}; // class SegmentedCloud

//=================================================================================================
//    SegmentedCloud public methods implementation
//=================================================================================================

template <typename PointCloudT>
Id SegmentedCloud::addSegment(const pcl::PointIndices& segment_to_add,
                              const PointCloudT& reference_cloud,
                              Id segment_id) {

  // This is the largest integer that can be reliably saved into a float.
  constexpr int max_segment_id = int(2.0f / std::numeric_limits<float>::epsilon());

  // Get and store segment ID in the intensity channel.
  if (segment_id <= 0) {
    segment_id = getNextId();
  }
  if (segment_id >= max_segment_id) {
    LOG_IF(ERROR, !g_too_many_segments_to_store_ids_in_intensity) <<
        "Segment ID being passed to point intensity is larger than float "
        "values allow. From this point on, intensity will no longer be usable "
        "to store segment IDs";
    g_too_many_segments_to_store_ids_in_intensity = true;
  }

  // Create the segment.
  Segment& segment = valid_segments_[segment_id];
  if (segment.empty()) {
    // If the segment is new.
    segment.segment_id = segment_id;
    segment.views.push_back(SegmentView());
  } else if (!keep_only_last_view_) {
    //TODO RD check whether that can be done in a nicer way.
    Features last_features = segment.getLastView().features;
    unsigned int last_semantic = segment.getLastView().semantic;
    unsigned int n_points_when_last_described = segment.getLastView().n_points_when_last_described;
    PointCloud last_reconstruction = segment.getLastView().reconstruction;
    segment.views.push_back(SegmentView());
    segment.getLastView().features = last_features;
    segment.getLastView().semantic = last_semantic;
    segment.getLastView().n_points_when_last_described = n_points_when_last_described;
    segment.getLastView().reconstruction = last_reconstruction;

    // TODO RD remove if compressing reconstruction not needed.
    // Compressed reconstruction.
    /*segment.getLastView().reconstruction_compressed.clear();
    segment.getLastView().reconstruction_compressed.reserve(last_reconstruction.points.size() / publish_every_x_points);
    unsigned int i = 0;
    for (const auto& point : last_reconstruction.points) {
      if (i % publish_every_x_points == 0) {
          segment.getLastView().reconstruction_compressed.points.emplace_back(point.x, point.y, point.z);
      }
      ++i;
    }*/
  }

  // Copy points into segment.
  segment.getLastView().point_cloud.clear();
  segment.getLastView().point_cloud.reserve(segment_to_add.indices.size());
  
  segment.getLastView().point_cloud_to_publish.clear();
  segment.getLastView().point_cloud_to_publish.reserve(segment_to_add.indices.size() / publish_every_x_points);
  unsigned int i = 0;
  for (const auto& index : segment_to_add.indices) {
    CHECK_LT(index, reference_cloud.points.size()) <<
        "Index is larger than the reference cloud size when adding segments. "
        "Check that the given reference cloud corresponds to the cluster "
        "indices.";

    // Store point inside the segment.
    segment.getLastView().point_cloud.points.emplace_back(reference_cloud.points[index].x,
                                                          reference_cloud.points[index].y,
                                                          reference_cloud.points[index].z);
    if (i % publish_every_x_points == 0) {
        segment.getLastView().point_cloud_to_publish.points.emplace_back(reference_cloud.points[index].x,
                                                            reference_cloud.points[index].y,
                                                            reference_cloud.points[index].z);
    }
    ++i;
  }

  segment.getLastView().calculateCentroid();
  return segment.segment_id;
}

} // namespace segmatch

#endif // SEGMATCH_SEGMENTED_CLOUD_HPP_
