#include "segmatch/segmented_cloud.hpp"

#include <utility>

#include <laser_slam/benchmarker.hpp>

namespace segmatch {

extern bool g_too_many_segments_to_store_ids_in_intensity(false);

void SegmentView::calculateCentroid() {
  centroid = segmatch::calculateCentroid(point_cloud);
}

/// \brief Generates a new Id number. Overall, no two valid segments should have the same Id.
Id SegmentedCloud::getNextId(const Id& begin_counting_from_this_id) {
  if (begin_counting_from_this_id == 0) {
    // Normal behavior.
    return ++current_id_;
  } else {
    // Used when you want to start counting segments at an arbitrary number.
    CHECK(current_id_ == 0) <<
        "Initializing the current_id after Ids have been assigned is forbidden. " <<
        "Check that the current_id is only initialized at the start of your code," <<
        "Otherwise collisions can occur.";
    current_id_ = begin_counting_from_this_id;
    LOG(INFO) << "Initialized the segment ids. Counting begins at id " << current_id_ << ".";
    return 0;
  }
}

void SegmentedCloud::addSegmentedCloud(
    const SegmentedCloud& segmented_cloud_to_add,
    const std::vector<std::pair<Id, Id>>& renamed_segments) {
  for (const auto& id_segment : segmented_cloud_to_add) {
    addValidSegment(id_segment.second);
  }

  for (const auto& renamed_segment : renamed_segments) {
    // Find segments affected by renaming
    Segment& deleted_segment = valid_segments_[renamed_segment.first];
    Segment& final_segment = valid_segments_[renamed_segment.second];

    // TODO How should we deal with the view of the segment being renamed when keeping multiple
    // views?
    // If necessary, add the deleted segment as a view of the final one.
    // if (!keep_only_last_view_ && !deleted_segment.views.empty()) {
    //   final_segment.views.push_back(deleted_segment.views.front());
    // }

    // Delete the old segment
    valid_segments_.erase(renamed_segment.first);
  }
  cleanEmptySegments(); 
}

void SegmentedCloud::addValidSegment(const Segment& segment_to_add) {
  CHECK(segment_to_add.hasValidId());
  //TODO(Renaud) unexpected behavior if the segment had more than one view.
  CHECK_EQ(segment_to_add.views.size(), 1u);
  Segment& segment_in_cloud = valid_segments_[segment_to_add.segment_id];
  if (keep_only_last_view_ || segment_in_cloud.empty()) {
    segment_in_cloud = segment_to_add;
  } else  if (static_cast<double>(segment_to_add.views[0u].point_cloud.size()) >=
      min_change_to_add_new_view * static_cast<double>(
          segment_in_cloud.getLastView().point_cloud.size())) {
    segment_in_cloud.views.push_back(segment_to_add.views[0u]);
  } else {
    segment_in_cloud.getLastView().features = segment_to_add.views[0u].features;
    segment_in_cloud.getLastView().semantic = segment_to_add.views[0u].semantic;
    segment_in_cloud.getLastView().centroid = segment_to_add.views[0u].centroid;
    segment_in_cloud.getLastView().timestamp_ns = segment_to_add.views[0u].timestamp_ns;
    segment_in_cloud.getLastView().T_w_linkpose = segment_to_add.views[0u].T_w_linkpose;
    segment_in_cloud.getLastView().n_occupied_voxels =
        segment_to_add.views[0u].n_occupied_voxels;
    segment_in_cloud.getLastView().n_points_when_last_described =
        segment_to_add.views[0u].n_points_when_last_described;
  }
  cleanEmptySegments();
}

size_t SegmentedCloud::getNumberOfValidSegments() const {
  return valid_segments_.size();
}

bool SegmentedCloud::findValidSegmentById(const Id segment_id, Segment* result) const {
  if (valid_segments_.count(segment_id)) {
    if (result != NULL) {
      *result = valid_segments_.at(segment_id);
    }
    return true;
  } else {
    return false;
  }
}

bool SegmentedCloud::findValidSegmentPtrById(const Id segment_id, Segment** result) {
  if (!findValidSegmentById(segment_id, NULL)) { return false; }
  if (result != NULL) { *result = CHECK_NOTNULL(&valid_segments_.at(segment_id)); }
  return true;
}

void SegmentedCloud::deleteSegmentsById(const std::vector<Id>& ids, size_t* n_removals) {
  if (n_removals != NULL) {
    *n_removals = 0;
  }
  for (const auto& id: ids) {
    valid_segments_.erase(id);
    // TODO obsolete?
    if (n_removals != NULL) {
      (*n_removals)++;
    }
  }
}

void SegmentedCloud::deleteSegmentsExcept(const std::vector<Id>& segment_ids_to_keep) {
  std::vector<Id> segment_ids_to_remove;
  for (const auto& id_segment : valid_segments_) {
    if (std::find(segment_ids_to_keep.begin(), segment_ids_to_keep.end(),
                  id_segment.first) == segment_ids_to_keep.end()) {
      segment_ids_to_remove.push_back(id_segment.first);
    }
  }
  deleteSegmentsById(segment_ids_to_remove);
  cleanEmptySegments();
}


void SegmentedCloud::calculateSegmentCentroids() {
  for (auto& id_segment : valid_segments_) {
    for (auto& view : id_segment.second.views) {
      view.calculateCentroid();
    }
  }
}

void SegmentedCloud::clear() {
  //TODO(Daniel): fill this function
  valid_segments_.clear();
}

PointCloud SegmentedCloud::centroidsAsPointCloud(
    std::vector<Id>& segment_id_for_each_point) const {

  PointCloud result;
  result.width = getNumberOfValidSegments();
  result.height = 1;
  for (const auto& id_segment: valid_segments_) {
      
    if(!id_segment.second.empty()) {
    PclPoint new_point = id_segment.second.getLastView().centroid;
    result.points.push_back(new_point);
    // Export the segment ids if desired.
    
    segment_id_for_each_point.push_back(id_segment.first);
    } else {
        LOG(INFO) << "was empty";
    }
  }
  result.width = result.points.size();


    CHECK(segment_id_for_each_point.size() == result.size()) <<
        "Each point should have a single segment id.";
  
  return result;
}

PointCloud SegmentedCloud::centroidsAsPointCloud(
    const laser_slam::SE3& center, double maximum_linkpose_distance,
    std::vector<Id>* segment_id_for_each_point_ptr) const {
  if (segment_id_for_each_point_ptr != NULL) {
    segment_id_for_each_point_ptr->clear();
  }
  PointCloud result;
  for (const auto& id_segment: valid_segments_) {
    Segment segment = id_segment.second;
    if (laser_slam::distanceBetweenTwoSE3(segment.getLastView().T_w_linkpose, center) <=
        maximum_linkpose_distance) {
      result.push_back(segment.getLastView().centroid);
      // Export the segment ids if desired.
      if (segment_id_for_each_point_ptr != NULL) {
        segment_id_for_each_point_ptr->push_back(segment.segment_id);
      }
    }
  }
  if (segment_id_for_each_point_ptr != NULL) {
    CHECK(segment_id_for_each_point_ptr->size() == result.size()) <<
        "Each point should have a single segment id.";
  }
  LOG(INFO) << "centroidsAsPointCloud returning " << result.size() << " from " <<
      valid_segments_.size() << " segments.";
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
  PointCloud centroid_cloud = this->centroidsAsPointCloud(segment_ids);

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

void SegmentedCloud::setTimeStampOfSegments(const laser_slam::Time& timestamp_ns) {
  for (auto& id_segment: valid_segments_) {
    id_segment.second.getLastView().timestamp_ns = timestamp_ns;
  }
}

void SegmentedCloud::setLinkPoseOfSegments(const laser_slam::SE3& link_pose) {
  for (auto& id_segment: valid_segments_) {
    id_segment.second.getLastView().T_w_linkpose = link_pose;
  }
}

void SegmentedCloud::setTrackId(unsigned int track_id) {
  for (auto& id_segment: valid_segments_) {
    id_segment.second.track_id = track_id;
  }
}

void SegmentedCloud::updateSegments(const std::vector<laser_slam::Trajectory>& trajectories) {
  for (auto& id_segment: valid_segments_) {
    SE3 new_pose = trajectories.at(id_segment.second.track_id).at(id_segment.second.getLastView().timestamp_ns);

    SE3 transformation = new_pose * id_segment.second.getLastView().T_w_linkpose.inverse();
    // Transform the point cloud.
    transformPointCloud(transformation, &id_segment.second.getLastView().point_cloud);

    // Transform the reconstruction.
    transformPointCloud(transformation, &id_segment.second.getLastView().reconstruction);
    transformPointCloud(transformation, &id_segment.second.getLastView().reconstruction_compressed);

    // Transform the segment centroid.
    transformPclPoint(transformation, &id_segment.second.getLastView().centroid);

    // Update the link pose.
    id_segment.second.getLastView().T_w_linkpose = new_pose;
  }
}

size_t SegmentedCloud::getCloseSegmentPairsCount(const float max_distance) const {
  size_t num_close_segments = 0u;
  float min_distance = std::numeric_limits<float>::max();
  for (const auto& segment_1 : valid_segments_) {
    for (const auto& segment_2 : valid_segments_) {
      if (segment_1.first < segment_2.first) {
        float distance = (
            segment_1.second.getLastView().centroid.getVector3fMap() -
            segment_2.second.getLastView().centroid.getVector3fMap()).norm();
        min_distance = std::min(min_distance, distance);
        if (distance <= max_distance) ++num_close_segments;
      }
    }
  }

  LOG(INFO) << "Minimum segment distance " << min_distance;
  BENCHMARK_RECORD_VALUE("SegmentedCloud.MinSegmentsDistance", min_distance);
  BENCHMARK_RECORD_VALUE("SegmentedCloud.NumCloseSegments", num_close_segments);
  return num_close_segments;
}

void SegmentedCloud::cleanEmptySegments() {
  std::vector<Id> ids_to_remove;
  for (const auto& segment : valid_segments_) {
      if (segment.second.empty()) ids_to_remove.push_back(segment.first);
  }
  deleteSegmentsById(ids_to_remove);
}

Id SegmentedCloud::current_id_ = 0;

} // namespace segmatch
