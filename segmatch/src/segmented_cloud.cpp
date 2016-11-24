#include "segmatch/segmented_cloud.hpp"

#include <cfenv>

namespace segmatch {

extern bool g_too_many_segments_to_store_ids_in_intensity(false);

IdOccurences::IdOccurences() {}

void IdOccurences::clear() {
  /* TODO: make sure to delete all Id_counters allocated with new */
  occurences_.clear();
}

size_t IdOccurences::size() const {
  return occurences_.size();
}

void IdOccurences::addOccurence(Id id_to_add) {
  bool found_in_occurences = false;
  if (id_to_add == kNoId) { return; }
  // If id_to_add is in the list, increment it.
  for (size_t i = 0u; i < occurences_.size(); ++i) {
    if (id_to_add == occurences_[i].id) {
      found_in_occurences = true;
      occurences_[i].count++;
      break;
    }
  }
  // If id_to_add is not in the occurences, add it.
  if (!found_in_occurences) {
    IdCounter* temp = new IdCounter;
    temp->id = id_to_add;
    temp->count = 1;
    occurences_.push_back(*temp);
  }
}

IdCounter IdOccurences::findHighestOccurence() const {
  // Reinitialize highest_occurence.
  IdCounter highest_occurence;
  highest_occurence.id = kNoId;
  highest_occurence.count = 0;
  // Find highest occurence and store it.
  for (size_t i = 0u; i < occurences_.size(); ++i) {
    if (occurences_[i].count > highest_occurence.count) {
      highest_occurence.id = occurences_[i].id;
      highest_occurence.count = occurences_[i].count;
    }
  }
  return highest_occurence;
}

void Segment::calculateCentroid() {
  std::feclearexcept(FE_ALL_EXCEPT);
  // Find the mean position of a segment.
  double x_mean = 0.0;
  double y_mean = 0.0;
  double z_mean = 0.0;
  const size_t n_points = point_cloud.points.size();
  for (size_t i = 0u; i < n_points; ++i) {
    x_mean += point_cloud.points[i].x / n_points;
    y_mean += point_cloud.points[i].y / n_points;
    z_mean += point_cloud.points[i].z / n_points;
  }

  centroid = PclPoint(x_mean, y_mean, z_mean);

  // Check that there were no overflows, underflows, or invalid float operations.
  if (std::fetestexcept(FE_OVERFLOW)) {
    LOG(ERROR) << "Overflow error in centroid computation.";
  } else if (std::fetestexcept(FE_UNDERFLOW)) {
    LOG(ERROR) << "Underflow error in centroid computation.";
  } else if (std::fetestexcept(FE_INVALID)) {
    LOG(ERROR) << "Invalid Flag error in centroid computation.";
  } else if (std::fetestexcept(FE_DIVBYZERO)) {
    LOG(ERROR) << "Divide by zero error in centroid computation.";
  }
}

/// \brief Generates a new Id number. Overall, no two valid segments should have the same Id.
Id SegmentedCloud::getNextId(const Id& begin_counting_from_this_id) {
  static Id current_id = 0;
  if (begin_counting_from_this_id == 0) {
    // Normal behavior.
    return ++current_id;
  } else {
    // Used when you want to start counting segments at an arbitrary number.
    CHECK(current_id == 0) <<
        "Initializing the current_id after Ids have been assigned is forbidden. " <<
        "Check that the current_id is only initialized at the start of your code," <<
        "Otherwise collisions can occur.";
    current_id = begin_counting_from_this_id;
    LOG(INFO) << "Initialized the segment ids. Counting begins at id " << current_id << ".";
    return 0;
  }
}

void SegmentedCloud::addSegmentedCloud(const SegmentedCloud& segmented_cloud_to_add) {
  if (!segmented_cloud_to_add.empty()) {
    for (size_t i = 0u; i < segmented_cloud_to_add.getNumberOfValidSegments(); ++i) {
      addValidSegment(segmented_cloud_to_add.getValidSegmentByIndex(i));
    }
  }
}

void SegmentedCloud::addSegmentsTo(const pcl::IndicesClusters& segments_to_add,
                                   const pcl::PointCloud<pcl::PointNormal>& reference_cloud,
                                   std::vector<Segment>* target_cluster_ptr,
                                   const bool also_copy_normals_data) {
  CHECK_NOTNULL(target_cluster_ptr);
  // Loop over clusters.
  for (size_t i = 0u; i < segments_to_add.size(); ++i) {
    std::vector<int> indices = segments_to_add[i].indices;

    // Create the segment.
    Segment segment;

    // Assign the segment an id.
    if (target_cluster_ptr == &valid_segments_) {
      segment.segment_id = getNextId();
    } else {
      segment.segment_id = kInvId;
    }

    // Copy points into segment.
    for (size_t j = 0u; j < indices.size(); ++j) {
      const size_t index = indices[j];
      CHECK_LT(index, reference_cloud.points.size()) <<
          "Indice is larger than the reference cloud size when adding segments. " <<
          "Check that the given reference cloud corresponds to the cluster indices.";
      pcl::PointNormal reference_point = reference_cloud.points[index];

      // Store point inside the segment.
      PointI point;
      point.x = reference_point.x;
      point.y = reference_point.y;
      point.z = reference_point.z;
      segment.point_cloud.push_back(point);

      // Copy DoN and normals into segment.
      if (also_copy_normals_data) {
        // Store normals.
        pcl::Normal normal;
        normal.normal_x = reference_point.normal_x;
        normal.normal_y = reference_point.normal_y;
        normal.normal_z = reference_point.normal_z;
        normal.curvature = reference_point.curvature;
        segment.normals.push_back(normal);
      }
    }

    // Store segment ID in the intensity channel.
    // 16777216 is the largest int that can be cast to float and back.
    if (segment.segment_id >= 16777216) {
      LOG_IF(ERROR, !g_too_many_segments_to_store_ids_in_intensity) <<
          "Segment Id being passed to point intensity is larger than float values " <<
          "allow. from this point on, intensity will no longer be usable to store segment ids";
      g_too_many_segments_to_store_ids_in_intensity = true;
    }
    for (size_t j = 0u; j < segment.point_cloud.size(); ++j) {
      segment.point_cloud.points[j].intensity = segment.segment_id;
    }

    segment.calculateCentroid();
    target_cluster_ptr->push_back(segment);
  }
}

void SegmentedCloud::addValidSegments(const pcl::IndicesClusters& segments_to_add,
                                      const pcl::PointCloud<pcl::PointNormal>& reference_cloud) {
  addSegmentsTo(segments_to_add, reference_cloud, &valid_segments_);
}

void SegmentedCloud::addSmallSegments(const pcl::IndicesClusters& segments_to_add,
                                      const pcl::PointCloud<pcl::PointNormal>& reference_cloud) {
  addSegmentsTo(segments_to_add, reference_cloud, &small_segments_);
}

void SegmentedCloud::addLargeSegments(const pcl::IndicesClusters& segments_to_add,
                                      const pcl::PointCloud<pcl::PointNormal>& reference_cloud) {
  addSegmentsTo(segments_to_add, reference_cloud, &large_segments_);
}

void SegmentedCloud::addValidSegment(const Segment& segment_to_add) {
  CHECK(segment_to_add.hasValidId());
  valid_segments_.push_back(segment_to_add);
}

size_t SegmentedCloud::getNumberOfValidSegments() const {
  return valid_segments_.size();
}

bool SegmentedCloud::findValidSegmentById(const Id segment_id, Segment* result,
                                          size_t* index_ptr) const {
  for (size_t i = 0u; i < getNumberOfValidSegments(); ++i) {
    if (getValidSegmentByIndex(i).segment_id == segment_id) {
      if (result != NULL) {
        *result = valid_segments_.at(i);
      }
      // If desired, pass the segment's index.
      if (index_ptr != NULL) {
        *index_ptr = i;
      }
      return true;
    }
  }
  return false;
}

bool SegmentedCloud::findValidSegmentPtrById(const Id segment_id, Segment** result,
                                             size_t* index_ptr) {
  // Find the index of the segment using already-existing code.
  size_t index;
  if (!findValidSegmentById(segment_id, NULL, &index)) { return false; }
  // If desired, pass the index.
  if (index_ptr != NULL) { *index_ptr = index; }
  // If desired, pass the segment ptr.
  if (result != NULL) { *result = CHECK_NOTNULL(&valid_segments_[index]); }
  return true;
}

Segment SegmentedCloud::getValidSegmentByIndex(const size_t index) const {
  CHECK_LT(index, valid_segments_.size()) <<
      "Attempted to access index out of bounds of valid_segments_";
  return valid_segments_[index];
}

Segment* SegmentedCloud::getValidSegmentPtrByIndex(const size_t index) {
  CHECK_LT(index, valid_segments_.size()) <<
      "Attempted to access index out of bounds of valid_segments_";
  return CHECK_NOTNULL(&valid_segments_[index]);
}

void SegmentedCloud::deleteSegmentsById(const std::vector<Id>& ids, size_t* n_removals) {
  if (n_removals != NULL) { *n_removals = 0; }
  for (size_t i = 0u; i < ids.size(); ++i) {
    size_t index;
    if (findValidSegmentById(ids.at(i), NULL, &index)) {
      valid_segments_.erase(valid_segments_.begin() + index);
      if (n_removals != NULL) { (*n_removals)++; }
    }
  }
}

void SegmentedCloud::calculateSegmentCentroids() {
  for (size_t i = 0u; i < getNumberOfValidSegments(); ++i) {
    getValidSegmentPtrByIndex(i)->calculateCentroid();
  }
}

void SegmentedCloud::clear() {
  //TODO(Daniel): fill this function
  valid_segments_.clear();
  small_segments_.clear();
  large_segments_.clear();
}

PointICloud SegmentedCloud::validSegmentsAsPointCloud(
    std::vector<Id>* segment_id_for_each_point_ptr) const {
  if (segment_id_for_each_point_ptr != NULL) { segment_id_for_each_point_ptr->clear(); }
  PointICloud result;
  for (size_t i = 0u; i < getNumberOfValidSegments(); ++i) {
    result += getValidSegmentByIndex(i).point_cloud;
    // Export the segment ids if desired.
    if (segment_id_for_each_point_ptr != NULL) {
      for (size_t j = 0u; j < getValidSegmentByIndex(i).point_cloud.size(); ++j) {
        segment_id_for_each_point_ptr->push_back(getValidSegmentByIndex(i).segment_id);
      }
      CHECK(segment_id_for_each_point_ptr->size() == result.size()) <<
          "Each point should have a single segment id.";
    }
  }
  return result;
}

PointICloud SegmentedCloud::validSegmentsAsPointCloudFromIds(
    const std::vector<Id>& ids, std::vector<Id>* segment_id_for_each_point_ptr) const {
  if (segment_id_for_each_point_ptr != NULL) {
    segment_id_for_each_point_ptr->clear();
  }
  PointICloud cloud;
  for (size_t i = 0u; i < ids.size(); ++i) {
    Segment segment;
    CHECK(findValidSegmentById(ids[i], &segment)) << "Segment not found.";
    cloud += segment.point_cloud;
    // Export the segment ids if desired.
    if (segment_id_for_each_point_ptr != NULL) {
      for (size_t j = 0u; j < segment.point_cloud.size(); ++j) {
        segment_id_for_each_point_ptr->push_back(segment.segment_id);
      }
      CHECK(segment_id_for_each_point_ptr->size() == cloud.size()) <<
          "Each point should have a single segment id.";
    }
  }
  return cloud;
}

PointCloud SegmentedCloud::centroidsAsPointCloud(
    std::vector<Id>* segment_id_for_each_point_ptr) const {
  if (segment_id_for_each_point_ptr != NULL) { segment_id_for_each_point_ptr->clear(); }
  PointCloud result;
  for (size_t i = 0u; i < getNumberOfValidSegments(); ++i) {
    result.push_back(getValidSegmentByIndex(i).centroid);
    // Export the segment ids if desired.
    if (segment_id_for_each_point_ptr != NULL) {
      segment_id_for_each_point_ptr->push_back(getValidSegmentByIndex(i).segment_id);
    }
  }
  if (segment_id_for_each_point_ptr != NULL) {
    CHECK(segment_id_for_each_point_ptr->size() == result.size()) <<
        "Each point should have a single segment id.";
  }
  return result;
}

bool SegmentedCloud::findNearestSegmentsToPoint(const PclPoint& point,
                                                unsigned int n_nearest_segments,
                                                double maximum_centroid_distance_m,
                                                std::vector<Id>* ids,
                                                std::vector<double>* distances) const {
  CHECK_NOTNULL(ids)->clear();
  CHECK_NOTNULL(distances)->clear();

  // Get centroid cloud with ids.
  std::vector<Id> segment_ids;
  PointCloud centroid_cloud = this->centroidsAsPointCloud(&segment_ids);

  //TODO deal properly with that case.
  //  CHECK_GE(segment_ids.size(), n_nearest_segments);

  if (segment_ids.size() >= n_nearest_segments) {
    // Set up nearest neighbour search.
    pcl::KdTreeFLANN<PclPoint> kdtree;
    PointCloudPtr centroid_cloud_ptr(new PointCloud);
    pcl::copyPointCloud(centroid_cloud, *centroid_cloud_ptr);
    kdtree.setInputCloud(centroid_cloud_ptr);

    std::vector<int> nearest_neighbour_indices(n_nearest_segments);
    std::vector<float> nearest_neighbour_squared_distances(n_nearest_segments);

    // Find the nearest neighbours.
    if (kdtree.nearestKSearch(point, n_nearest_segments, nearest_neighbour_indices,
                              nearest_neighbour_squared_distances) <= 0) {
      LOG(ERROR) << "Nearest neighbour search failed.";
      return false;
    }

    for (unsigned int i = 0u; i < n_nearest_segments; ++i) {
      if (nearest_neighbour_squared_distances[i] <= maximum_centroid_distance_m) {
        ids->push_back(segment_ids[nearest_neighbour_indices[i]]);
        distances->push_back(nearest_neighbour_squared_distances[i]);
      }
    }
  }

  return true;
}

/*
 * \brief Find which segments overlap.
 * Two points overlap if they are within 'overlap_radius' of each other.
 * overlap_radius's unit is the same as the point cloud (meters, millimeters)
 */

//TODO to speed up that expensive function, we could, for each segment in the source cloud,
// find the N (=3?) closest segments in the target cloud, by comparing the distance of the centroids.
// Then build a cloud from these N closest segment and do the nearest neighbours on this.
// See functions findClosestSegmentsByCentroid and validSegmentsAsPointCloudFromIds.
bool SegmentedCloud::computeSegmentOverlaps(const SegmentedCloud& target_segmented_cloud,
                                            const float overlap_radius,
                                            const unsigned int number_nearest_segments,
                                            double maximum_centroid_distance_m,
                                            Overlaps* overlaps) const {
  LOG(INFO) << "Computing segment overlaps.";
  CHECK_NOTNULL(overlaps)->clear();
  PointICloud target_cloud = target_segmented_cloud.validSegmentsAsPointCloud();
  if (validSegmentsAsPointCloud().empty() || target_cloud.empty()) {
    LOG(WARNING) << "One of the segmented cloud is empty. Found no overlaps.";
    return false;
  }

  // Initialize the detailed vector of offenders for each segments.
  overlaps->detailed_offenders_.resize(getNumberOfValidSegments());

  // Loop over valid segments in this SegmentedCloud.
  const float overlap_radius_squared = overlap_radius * overlap_radius;
  for (size_t i = 0u; i < getNumberOfValidSegments(); ++i) {
    Segment current_segment = getValidSegmentByIndex(i);
    IdOccurences id_occurences_in_segment_overlap;

    // Find the nearest segments by centroid.
    std::vector<Id> nearest_segments_ids;
    std::vector<double> nearest_segments_distances;
    target_segmented_cloud.findNearestSegmentsToPoint(current_segment.centroid,
                                                      number_nearest_segments,
                                                      maximum_centroid_distance_m,
                                                      &nearest_segments_ids,
                                                      &nearest_segments_distances);

    std::vector<Id> nearest_segments_cloud_segment_ids;
    PointICloud nearest_segments_cloud = target_segmented_cloud.validSegmentsAsPointCloudFromIds(
        nearest_segments_ids, &nearest_segments_cloud_segment_ids);

    // Loop over every point in the segment.
    unsigned int n_tested_points = 0;
    for (size_t j = 0u; j < current_segment.point_cloud.size(); j += 10) {
      PointI point = current_segment.point_cloud.at(j);

      Id segment_id_of_nearest_neighbour = kNoId;

      // Find the nearest neighbours in nearest_segments_cloud if it is not empty due to
      // parameter maximum_centroid_distance_m.
      if (nearest_segments_cloud.size() > 0u) {
        size_t nn_index;
        float squared_distance;
        CHECK(findNearestNeighbour(point, nearest_segments_cloud, &nn_index, &squared_distance));

        // Ignore the nearest neighbour if it is outside overlap_radius.
        if (squared_distance <= overlap_radius_squared) {
          CHECK_LT(static_cast<size_t>(nn_index),
                   nearest_segments_cloud.points.size());
          segment_id_of_nearest_neighbour  = nearest_segments_cloud_segment_ids.at(nn_index);
          id_occurences_in_segment_overlap.addOccurence(segment_id_of_nearest_neighbour);
        }
      }


      // Store the overlapping point's segment ID.
      overlaps->detailed_offenders_.at(i).push_back(segment_id_of_nearest_neighbour);

      ++n_tested_points;
    }

    // Find and store current segment's worst offender.
    IdCounter worst_offender = id_occurences_in_segment_overlap.findHighestOccurence();
    worst_offender.total_count = n_tested_points;
    overlaps->worst_offenders_.push_back(worst_offender);

    // Logging.
    if (worst_offender.id != kNoId) {
      LOG(INFO) << "Worst offender for " << current_segment.segment_id << ": "
          << worst_offender.id << " with "
          << 100.0 * worst_offender.count/worst_offender.total_count
          << "% (" << worst_offender.count << "/"
          << worst_offender.total_count << ") overlapping points";
    } else {
      LOG(INFO) << "Worst offender for " << current_segment.segment_id << ": None.";
    }
  }

  CHECK(overlaps->worst_offenders_.size() == getNumberOfValidSegments());
  CHECK(overlaps->detailed_offenders_.size() == getNumberOfValidSegments());
  return true;
}

void SegmentedCloud::setTimeStampOfSegments(const laser_slam::Time& timestamp_ns) {
  for (Segment& segment: valid_segments_) {
    segment.timestamp_ns = timestamp_ns;
  }
}

void SegmentedCloud::setLinkPoseOfSegments(const laser_slam::SE3& link_pose) {
  for (Segment& segment: valid_segments_) {
    segment.T_w_linkpose = link_pose;
  }
}

void SegmentedCloud::setTrackId(unsigned int track_id) {
  for (Segment& segment: valid_segments_) {
    segment.track_id = track_id;
  }
}

void SegmentedCloud::updateSegments(const laser_slam::Trajectory& trajectory) {
  for (Segment& segment: valid_segments_) {
    SE3 new_pose = trajectory.at(segment.timestamp_ns);

    SE3 transformation = new_pose * segment.T_w_linkpose.inverse();
    // Transform the point cloud.
    transformPointCloud(transformation, &segment.point_cloud);

    // Transform the segment centroid.
    transformPclPoint(transformation, &segment.centroid);

    // Update the link pose.
    segment.T_w_linkpose = new_pose;
  }
}

} // namespace segmatch
