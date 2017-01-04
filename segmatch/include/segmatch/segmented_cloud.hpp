#ifndef SEGMATCH_SEGMENTED_CLOUD_HPP_
#define SEGMATCH_SEGMENTED_CLOUD_HPP_

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "segmatch/common.hpp"
#include "segmatch/features.hpp"

namespace segmatch {

/// \brief Triggered when the amount of segments becomes too high.
extern bool g_too_many_segments_to_store_ids_in_intensity;

/*
 * IdOccurences are a list of id counts, where an Id can only appear once.
 */
class IdOccurences {
 public:
  IdOccurences();
  IdOccurences(const std::vector<Id>& id_list) {
    for (size_t i = 0u; i < id_list.size(); ++i) { addOccurence(id_list.at(i)); }
  }

  void clear();
  size_t size() const;
  void addOccurence(Id id_to_add);
  IdCounter findHighestOccurence() const;

 private:
  std::vector<IdCounter> occurences_;
}; // class IdOccurences

class Overlaps {
 public:
  Overlaps() {};
  void clear() {
    worst_offenders_.clear();
    detailed_offenders_.clear();
  }

  // For each segment, the segment id which overlaps it the most.
  std::vector<IdCounter> worst_offenders_;
  // For each segment, the segment ids overlapping at each point.
  std::vector<std::vector<Id> > detailed_offenders_;
}; // class Overlaps

struct Segment {
  Segment () {}
  bool empty() const { return point_cloud.empty(); }
  void clear() {
    segment_id = kNoId;
    point_cloud.clear();
    normals.clear();
    features.clear();
  }
  bool hasValidId() const { return segment_id != kNoId && segment_id != kInvId; }
  void calculateCentroid();

  Id segment_id = kNoId;
  /// Points
  PointICloud point_cloud;
  // Normals
  pcl::PointCloud<pcl::Normal> normals;
  // Overlap with other segments at each point.
  std::vector<Id> overlap;
  IdCounter most_overlap;
  // Features.
  Features features;
  PclPoint centroid;

  // Time at which the segment was created.
  laser_slam::Time timestamp_ns;

  // Trajectory pose to which the segment is linked.
  laser_slam::SE3 T_w_linkpose;
  unsigned int track_id;
};

class SegmentedCloud {
 public:
  SegmentedCloud() {};
  Id getNextId(const Id& begin_counting_from_this_id = 0);
  SegmentedCloud& operator+= (const SegmentedCloud& rhs) {
    for (auto& id_segment: rhs.valid_segments_) {
      this->addValidSegment(id_segment.second);
    }
    return *this;
  }
  void addSegmentedCloud(const SegmentedCloud& segmented_cloud_to_add);
  void addSegments(const pcl::IndicesClusters& segments_to_add,
                   const pcl::PointCloud<pcl::PointNormal>& reference_cloud,
                   const bool also_copy_normals_data=true);
  void addValidSegments(const pcl::IndicesClusters& segments_to_add,
                        const pcl::PointCloud<pcl::PointNormal>& reference_cloud);
  /// \brief Adds a copy of a segment to this cloud.
  ///        This causes two segments to exist with the same Id.
  void addValidSegment(const Segment& segment_to_add);
  size_t getNumberOfValidSegments() const;
  bool empty() const { return getNumberOfValidSegments() == 0; }
  bool findValidSegmentById(const Id segment_id, Segment* result) const;
  bool findValidSegmentPtrById(const Id segment_id, Segment** result);
  void deleteSegmentsById(const std::vector<Id>& ids, size_t* n_removals=NULL);
  bool computeSegmentOverlaps(const SegmentedCloud& target_segmented_cloud,
                              const float overlap_radius,
                              const unsigned int number_nearest_segments,
                              double maximum_centroid_distance_m,
                              Overlaps* overlaps) const;
  /// brief Clear the segmented cloud.
  void clear();
  void calculateSegmentCentroids();
  void transform(const Eigen::Matrix4f& transform_matrix) {
    for (auto& id_segment: valid_segments_) {
      pcl::transformPointCloud(id_segment.second.point_cloud,
                               id_segment.second.point_cloud,
                               transform_matrix);
    }
    calculateSegmentCentroids(); // TODO: is transforming the centroids faster?
  }
  SegmentedCloud transformed(const Eigen::Matrix4f& transform_matrix) const {
    SegmentedCloud self_copy = *this;
    self_copy.transform(transform_matrix);
    return self_copy;
  }
  PointICloud validSegmentsAsPointCloud(
      std::vector<Id>* segment_id_for_each_point_ptr=NULL) const;
  PointCloud centroidsAsPointCloud(
      std::vector<Id>* segment_id_for_each_point_ptr=NULL) const;

  // Get the centroids which linkpose falls within a radius.
  PointCloud centroidsAsPointCloud(const laser_slam::SE3& center,
                                   double maximum_linkpose_distance,
                                   std::vector<Id>* segment_id_for_each_point_ptr=NULL) const;

  bool findNearestSegmentsToPoint(const PclPoint& point,
                                  unsigned int n_closest_segments,
                                  double maximum_centroid_distance_m,
                                  std::vector<Id>* ids,
                                  std::vector<double>* distances) const;

  PointICloud validSegmentsAsPointCloudFromIds(
      const std::vector<Id>& ids, std::vector<Id>* segment_id_for_each_point_ptr) const;

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

 private:
  std::unordered_map<Id, Segment> valid_segments_;
}; // class SegmentedCloud

//TODO Move inside of SegmentedCloud? eg. validSegmentsAsColoredCloud().
static void segmentedCloudToCloud(const SegmentedCloud& segmented_cloud,
                                  PointICloud* cloud_out) {
  CHECK_NOTNULL(cloud_out);
  //TODO do this without creating a new object.
  PointICloud cloud;
  std::vector<int> permuted_indexes;
  for (unsigned int i = 0u; i < segmented_cloud.getNumberOfValidSegments(); ++i) {
    permuted_indexes.push_back(i);
  }
  std::random_shuffle(permuted_indexes.begin(), permuted_indexes.end());
  unsigned int i = 0u;
  for (std::unordered_map<Id, Segment>::const_iterator it = segmented_cloud.begin();
      it != segmented_cloud.end(); ++it) {
    PointICloud segment_cloud = it->second.point_cloud;
    // Color Segment Cloud.
    for (size_t j = 0u; j < segment_cloud.size(); ++j) {
      segment_cloud.points[j].intensity = permuted_indexes[i];
    }
    cloud += segment_cloud;
    ++i;
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();

  *cloud_out = cloud;
}

typedef std::pair<SegmentedCloud, SegmentedCloud> SegmentedCloudPair;

} // namespace segmatch

#endif // SEGMATCH_SEGMENTED_CLOUD_HPP_
