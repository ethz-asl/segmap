#include "segmatch/segmatch.hpp"

#include <laser_slam/common.hpp>
#include <pcl/recognition/cg/geometric_consistency.h>

namespace segmatch {

using namespace laser_slam;

SegMatch::SegMatch(const SegMatchParams& params) {
  init(params);
}

SegMatch::SegMatch() {
  LOG(INFO) << "Do not forget to initialize SegMatch.";
}

SegMatch::~SegMatch() {
  descriptors_.reset();
  segmenter_.reset();
}

void SegMatch::init(const SegMatchParams& params,
                    unsigned int num_tracks) {
  params_ = params;
  descriptors_ = std::unique_ptr<Descriptors>(new Descriptors(params.descriptors_params));
  segmenter_ = create_segmenter(params.segmenter_params);
  classifier_ = std::unique_ptr<OpenCvRandomForest>(
      new OpenCvRandomForest(params.classifier_params));

  // Create containers for the segmentation poses.
  CHECK_GT(num_tracks, 0u);
  for (unsigned int i = 0u; i < num_tracks; ++i) {
    segmentation_poses_.push_back(laser_slam::Trajectory());
  }
}

void SegMatch::setParams(const SegMatchParams& params) {
  LOG(INFO) << "Reseting segmatch's params.";
  params_ = params;
  classifier_->resetParams(params.classifier_params);
  LOG(INFO) << "GC resolution " << params_.geometric_consistency_params.resolution;
  LOG(INFO) << "GC min cluster size " << params_.geometric_consistency_params.min_cluster_size;
}

void SegMatch::processAndSetAsSourceCloud(const PointICloud& source_cloud,
                                          const laser_slam::Pose& latest_pose,
                                          const unsigned int track_id) {
  // Save the segmentation pose.
  segmentation_poses_[track_id][latest_pose.time_ns] = latest_pose.T_w;

  // Apply a cylindrical filter on the input cloud.
  PointICloud filtered_cloud = source_cloud;
  applyCylindricalFilter(laserSlamPoseToPclPoint(latest_pose),
                         params_.segmentation_radius_m,
                         kCylinderHeight_m, &filtered_cloud);

  // Segment the cloud and set segment information.
  segmenter_->segment(filtered_cloud, &segmented_source_cloud_);
  segmented_source_cloud_.setTimeStampOfSegments(latest_pose.time_ns);
  segmented_source_cloud_.setLinkPoseOfSegments(latest_pose.T_w);
  segmented_source_cloud_.setTrackId(track_id);

  // Filter the boundary segments.
  if (params_.filter_boundary_segments) {
    filterBoundarySegmentsOfSourceCloud(laserSlamPoseToPclPoint(latest_pose));
  }

  // Describe the cloud.
  descriptors_->describe(&segmented_source_cloud_);
}

void SegMatch::processAndSetAsTargetCloud(const PointICloud& target_cloud) {
  // Process the cloud.
  processCloud(target_cloud, &segmented_target_cloud_);

  // Overwrite the old target.
  classifier_->setTarget(segmented_target_cloud_);
}

void SegMatch::transferSourceToTarget() {
  target_queue_.push_back(segmented_source_cloud_);

  // Remove empty clouds from queue.
  std::vector<SegmentedCloud>::iterator it = target_queue_.begin();
  while(it != target_queue_.end()) {
    if(it->empty()) {
      it = target_queue_.erase(it);
    } else {
      ++it;
    }
  }

  // Check whether the pose linked to the segments of the oldest cloud in the queue
  // has a sufficient distance to the latest pose.
  bool try_adding_latest_cloud = true;
  unsigned int num_cloud_transfered = 0u;
  while (try_adding_latest_cloud && num_cloud_transfered < kMaxNumberOfCloudToTransfer) {
    try_adding_latest_cloud = false;
    if (!target_queue_.empty()) {
      // Get an iterator to the latest cloud with the same track_id.
      it = target_queue_.begin();
      bool found = false;
      while (!found && it != target_queue_.end()) {
        if (it->getValidSegmentByIndex(0u).track_id ==
            segmented_source_cloud_.getValidSegmentByIndex(0u).track_id) {
          found = true;
        } else {
          ++it;
        }
      }

      if (found) {
        // Check distance since last segmentation.
        laser_slam::SE3 oldest_queue_pose = it->getValidSegmentByIndex(0u).T_w_linkpose;
        laser_slam::SE3 latest_pose =
            segmented_source_cloud_.getValidSegmentByIndex(0u).T_w_linkpose;
        const double distance = distanceBetweenTwoSE3(oldest_queue_pose, latest_pose);
        if (distance > params_.segmentation_radius_m) {
          if (params_.filter_duplicate_segments) {
            filterDuplicateSegmentsOfTargetMap(*it);
          }
          segmented_target_cloud_.addSegmentedCloud(*it);
          target_queue_.erase(it);
          ++num_cloud_transfered;
          try_adding_latest_cloud = true;
          LOG(INFO) << "Transfered a source cloud to the target cloud.";
        }
      }
    }
  }

  if (num_cloud_transfered > 0u) {
    classifier_->setTarget(segmented_target_cloud_);
  }
}

void SegMatch::processCloud(const PointICloud& target_cloud,
                            SegmentedCloud* segmented_cloud,
                            std::vector<double>* timings) {
  laser_slam::Clock clock;
  segmenter_->segment(target_cloud, segmented_cloud);
  if (timings != NULL) {
    clock.takeTime();
    // First timing is segmentation.
    timings->push_back(clock.getRealTime());
  }

  std::vector<double> segmentation_timings;
  descriptors_->describe(segmented_cloud, &segmentation_timings);
  if (timings != NULL && !segmentation_timings.empty()) {
    // Following timings are description.
    for (size_t i = 0u; i < segmentation_timings.size(); ++i) {
      timings->push_back(segmentation_timings[i]);
    }
  }
}

PairwiseMatches SegMatch::findMatches(PairwiseMatches* matches_after_first_stage) {
  PairwiseMatches candidates;
  if (!segmented_source_cloud_.empty()) {
    candidates = classifier_->findCandidates(segmented_source_cloud_,
                                             matches_after_first_stage);
  }
  return candidates;
}

bool SegMatch::filterMatches(const PairwiseMatches& predicted_matches,
                             PairwiseMatches* filtered_matches_ptr,
                             RelativePose* loop_closure,
                             std::vector<PointICloudPair>* matched_segment_clouds) {
  if (matched_segment_clouds != NULL) { matched_segment_clouds->clear(); }

  PairwiseMatches filtered_matches;
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

  if (!predicted_matches.empty()) {
    //TODO: use a (gc) filtering class for an extra layer of abstraction?
    // Build point clouds out of the centroids for geometric consistency grouping.
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
    PointCloudPtr first_cloud(new PointCloud());
    PointCloudPtr second_cloud(new PointCloud());

    // Create clouds for geometric consistency.
    for (size_t i = 0u; i < predicted_matches.size(); ++i) {
      // First centroid.
      PclPoint first_centroid = predicted_matches.at(i).getCentroids().first;
      first_cloud->push_back(first_centroid);
      // Second centroid.
      PclPoint second_centroid = predicted_matches.at(i).getCentroids().second;
      second_cloud->push_back(second_centroid);
      float squared_distance = 1.0 - predicted_matches.at(i).confidence_;
      correspondences->push_back(pcl::Correspondence(i, i, squared_distance));
    }

    if (!correspondences->empty()) {
      // Perform geometric consistency grouping.
      RotationsTranslations correspondence_transformations;
      Correspondences clustered_corrs;
      pcl::GeometricConsistencyGrouping<PclPoint, PclPoint> geometric_consistency_grouping;
      geometric_consistency_grouping.setGCSize(params_.geometric_consistency_params.resolution);
      geometric_consistency_grouping.setGCThreshold(
          params_.geometric_consistency_params.min_cluster_size);
      geometric_consistency_grouping.setInputCloud(first_cloud);
      geometric_consistency_grouping.setSceneCloud(second_cloud);
      geometric_consistency_grouping.setModelSceneCorrespondences(correspondences);
      geometric_consistency_grouping.recognize(correspondence_transformations, clustered_corrs);

      if (!clustered_corrs.empty()) {
        // Find largest cluster.
        size_t largest_cluster_size = 0;
        size_t largest_cluster_index = 0;
        for (size_t i = 0u; i < clustered_corrs.size(); ++i) {
          if (clustered_corrs[i].size() >= largest_cluster_size) {
            largest_cluster_size = clustered_corrs[i].size();
            largest_cluster_index = i;
          }
        }

        // Catch the cases when PCL returns clusters smaller than the minimum cluster size.
        if (largest_cluster_size >= params_.geometric_consistency_params.min_cluster_size) {
          // Create pairwise matches from largest cluster
          pcl::Correspondences largest_cluster = clustered_corrs.at(largest_cluster_index);
          LOG(INFO) << "Returning the largest cluster at index " << largest_cluster_index
              << "...of size " << largest_cluster.size() << ".";
          for (size_t i = 0u; i < largest_cluster.size(); ++i) {
            // TODO: This assumes the matches from which the cloud was created
            //       are indexed in the same way as the cloud.
            //       (i.e. match[i] -> first_cloud[i] with second_cloud[i])
            //       Otherwise, this check will fail.
            CHECK(largest_cluster.at(i).index_query == largest_cluster.at(i).index_match);
            filtered_matches.push_back(predicted_matches.at(largest_cluster.at(i).index_query));
          }

          transformation = correspondence_transformations.at(largest_cluster_index);

          // Save the transformation.
          last_transformation_ = transformation;
        }
      }
    }

    // If desired, pass the filtered matches.
    if (filtered_matches_ptr != NULL && !filtered_matches.empty()) {
      *filtered_matches_ptr = filtered_matches;
    }

    // If desired, return the matched segments pointcloud.
    if (matched_segment_clouds != NULL && !filtered_matches.empty()) {
      LOG(INFO) << "Returning " << filtered_matches.size() << " matching segment clouds.";
      for (size_t i = 0u; i < filtered_matches.size(); ++i) {
        PointICloudPair cloud_pair;
        Segment segment;
        segmented_source_cloud_.findValidSegmentById(filtered_matches[i].ids_.first, &segment);

        for (size_t i = 0u; i < segment.point_cloud.size(); ++i) {
          segment.point_cloud[i].x -= segment.centroid.x;
          segment.point_cloud[i].y -= segment.centroid.y;
          segment.point_cloud[i].z -= segment.centroid.z;
        }

        cloud_pair.first = segment.point_cloud;
        segmented_target_cloud_.findValidSegmentById(
            filtered_matches[i].ids_.second, &segment);

        for (size_t i = 0u; i < segment.point_cloud.size(); ++i) {
          segment.point_cloud[i].x -= segment.centroid.x;
          segment.point_cloud[i].y -= segment.centroid.y;
          segment.point_cloud[i].z -= segment.centroid.z;
        }

        cloud_pair.second = segment.point_cloud;

        matched_segment_clouds->push_back(cloud_pair);
      }
    }

    // If desired, return the loop-closure.
    if (loop_closure != NULL && !filtered_matches.empty()) {
      // Find the trajectory poses to be linked by the loop-closure.
      // For each segment, find the timestamp of the closest segmentation pose.
      std::vector<Time> source_segmentation_times;
      std::vector<Time> target_segmentation_times;
      std::vector<Segment> source_segments;
      std::vector<Segment> target_segments;
      for (const auto& match: filtered_matches) {
        Segment segment;
        CHECK(segmented_source_cloud_.findValidSegmentById(match.ids_.first, &segment));
        source_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
        source_segments.push_back(segment);

        CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
        target_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
        target_segments.push_back(segment);
      }

      // Save the most occuring time stamps as timestamps for loop closure.
      loop_closure->time_a_ns = findMostOccuringTime(target_segmentation_times);
      loop_closure->time_b_ns = findMostOccuringTime(source_segmentation_times);

      CHECK(loop_closure->time_a_ns < loop_closure->time_b_ns);

      // Get the track_id of segments created at that time.
      bool found = false;
      for (size_t i = 0u; i < source_segments.size(); ++i) {
        if (!found && source_segmentation_times[i] == loop_closure->time_b_ns) {
          found = true;
          loop_closure->track_id_b = source_segments[i].track_id;
        }
      }
      CHECK(found);
      found = false;
      for (size_t i = 0u; i < target_segments.size(); ++i) {
        if (!found && target_segmentation_times[i] == loop_closure->time_a_ns) {
          found = true;
          loop_closure->track_id_a = target_segments[i].track_id;
        }
      }
      CHECK(found);

      SE3 w_T_a_b = fromApproximateTransformationMatrix(transformation);
      loop_closure->T_a_b = w_T_a_b;

      // Save the loop closure.
      loop_closures_.push_back(*loop_closure);
    }

    // Save a copy of the fitered matches.
    last_filtered_matches_ = filtered_matches;
  }


  return !filtered_matches.empty();
}

void SegMatch::update(const std::vector<laser_slam::Trajectory>& trajectories) {
  CHECK_EQ(trajectories.size(), segmentation_poses_.size());
  // Update the segmentation positions.
  for (size_t i = 0u; i < trajectories.size(); ++i) {
    for (auto& pose: segmentation_poses_[i]){
      pose.second = trajectories.at(i).at(pose.first);
    }
  }
  // Update the source, target and clouds in the buffer.
  segmented_source_cloud_.updateSegments(trajectories);
  segmented_target_cloud_.updateSegments(trajectories);
  for (auto& segmented_cloud: target_queue_) {
    segmented_cloud.updateSegments(trajectories);
  }

  // Update the last filtered matches.
  for (auto& match: last_filtered_matches_) {
    Segment segment;
    CHECK(segmented_source_cloud_.findValidSegmentById(match.ids_.first, &segment));
    match.centroids_.first = segment.centroid;
    CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
    match.centroids_.second = segment.centroid;
  }

  //TODO: filter duplicates.
}

void SegMatch::getSourceRepresentation(PointICloud* source_representation,
                                       const double& distance_to_raise) const {
  segmentedCloudToCloud(segmented_source_cloud_.transformed(
      Eigen::Affine3f(Eigen::Translation3f(0,0,distance_to_raise)).matrix()),
                        source_representation);

}

void SegMatch::getTargetRepresentation(PointICloud* target_representation) const {
  segmentedCloudToCloud(segmented_target_cloud_, target_representation);
}

void SegMatch::getTargetSegmentsCentroids(PointICloud* segments_centroids) const {
  CHECK_NOTNULL(segments_centroids);
  PointICloud cloud;
  std::vector<int> permuted_indexes;
  for (unsigned int i = 0u; i < segmented_target_cloud_.getNumberOfValidSegments(); ++i) {
    permuted_indexes.push_back(i);
  }
  std::random_shuffle(permuted_indexes.begin(), permuted_indexes.end());
  for (size_t i = 0u; i < segmented_target_cloud_.getNumberOfValidSegments(); ++i) {
    PointI centroid;
    Segment segment = segmented_target_cloud_.getValidSegmentByIndex(i);
    centroid.x = segment.centroid.x;
    centroid.y = segment.centroid.y;
    centroid.z = segment.centroid.z;
    centroid.intensity = permuted_indexes[i];
    cloud.points.push_back(centroid);
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();
  // TODO use move to to avoid deep copy.
  *segments_centroids = cloud;
}

void SegMatch::getSourceSegmentsCentroids(PointICloud* segments_centroids) const {
  // TODO combine with function above and reuse code.
  CHECK_NOTNULL(segments_centroids);
  PointICloud cloud;
  std::vector<int> permuted_indexes;
  for (unsigned int i = 0u; i < segmented_source_cloud_.getNumberOfValidSegments(); ++i) {
    permuted_indexes.push_back(i);
  }
  std::random_shuffle(permuted_indexes.begin(), permuted_indexes.end());
  for (size_t i = 0u; i < segmented_source_cloud_.getNumberOfValidSegments(); ++i) {
    PointI centroid;
    Segment segment = segmented_source_cloud_.getValidSegmentByIndex(i);
    centroid.x = segment.centroid.x;
    centroid.y = segment.centroid.y;
    centroid.z = segment.centroid.z;
    centroid.intensity = permuted_indexes[i];
    cloud.points.push_back(centroid);
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();
  // TODO use move to to avoid deep copy.
  *segments_centroids = cloud;
}

void SegMatch::getLoopClosures(std::vector<laser_slam::RelativePose>* loop_closures) const {
  CHECK_NOTNULL(loop_closures);
  *loop_closures = loop_closures_;
}

void SegMatch::getPastMatchesRepresentation(PointPairs* past_matches,
                                            PointPairs* invalid_past_matches) const {
  // TODO
}

void SegMatch::getLatestMatch(int64_t* time_a, int64_t* time_b,
                              Eigen::Matrix4f* transform_a_b,
                              std::vector<int64_t>* collector_times) const {
  // TODO
}

void SegMatch::filterBoundarySegmentsOfSourceCloud(const PclPoint& center) {
  if (!segmented_source_cloud_.empty()) {
    const double squared_radius = params_.boundary_radius_m * params_.boundary_radius_m;
    // Get a list of segments with at least one point outside the boundary.
    std::vector<Id> boundary_segments_ids;
    for (size_t i = 0u; i < segmented_source_cloud_.getNumberOfValidSegments(); ++i) {
      Segment segment = segmented_source_cloud_.getValidSegmentByIndex(i);
      PointICloud segment_cloud = segment.point_cloud;
      // Loop over points until one is found outside of the boundary.
      for (size_t j = 0u; j < segment_cloud.size(); ++j) {
        PointI point = segment_cloud.at(j);
        point.x -= center.x;
        point.y -= center.y;
        if ((point.x * point.x + point.y * point.y) >= squared_radius) {
          // If found, add the segment to the deletion list, and move on to the next segment.
          boundary_segments_ids.push_back(segment.segment_id);
          break;
        }
      }
    }

    // Remove boundary segments.
    size_t n_removals;
    segmented_source_cloud_.deleteSegmentsById(boundary_segments_ids, &n_removals);
    LOG(INFO) << "Removed " << n_removals << " boundary segments.";
  }
}

void SegMatch::filterDuplicateSegmentsOfTargetMap(const SegmentedCloud& cloud_to_be_added) {
  if (!cloud_to_be_added.empty()) {
    laser_slam::Clock clock;
    std::vector<Id> duplicate_segments_ids;
    std::vector<Id> target_segment_ids;
    PointCloud centroid_cloud = segmented_target_cloud_.centroidsAsPointCloud(&target_segment_ids);

    const unsigned int n_nearest_segments = 1u;
    if (target_segment_ids.size() > n_nearest_segments) {
      // Set up nearest neighbour search.
      pcl::KdTreeFLANN<PclPoint> kdtree;
      PointCloudPtr centroid_cloud_ptr(new PointCloud);
      pcl::copyPointCloud(centroid_cloud, *centroid_cloud_ptr);
      kdtree.setInputCloud(centroid_cloud_ptr);

      for (size_t i = 0u; i < cloud_to_be_added.getNumberOfValidSegments(); ++i) {
        std::vector<int> nearest_neighbour_indice(n_nearest_segments);
        std::vector<float> nearest_neighbour_squared_distance(n_nearest_segments);

        // Find the nearest neighbours.
        if (kdtree.nearestKSearch(cloud_to_be_added.getValidSegmentByIndex(i).centroid,
                                  n_nearest_segments, nearest_neighbour_indice,
                                  nearest_neighbour_squared_distance) <= 0) {
          LOG(ERROR) << "Nearest neighbour search failed.";
        }

        if (nearest_neighbour_squared_distance[0u] <= params_.centroid_distance_threshold_m) {
          duplicate_segments_ids.push_back(target_segment_ids[nearest_neighbour_indice[0u]]);
        }
      }
    }

    // Remove duplicates.
    size_t n_removals;
    segmented_target_cloud_.deleteSegmentsById(duplicate_segments_ids, &n_removals);
    clock.takeTime();
    LOG(INFO) << "Removed " << n_removals << " duplicate segments in " <<
        clock.getRealTime() << " ms.";
  }
}

Time SegMatch::findTimeOfClosestSegmentationPose(const Segment& segment) const {
  const Time segment_time_ns = segment.timestamp_ns;

  // Create the time window for which to consider poses.
  Time min_time_ns;
  if (segment_time_ns < kMaxTimeDiffBetweenSegmentAndPose_ns) {
    min_time_ns = 0u;
  } else {
    min_time_ns = segment_time_ns - kMaxTimeDiffBetweenSegmentAndPose_ns;
  }
  const Time max_time_ns = segment_time_ns + kMaxTimeDiffBetweenSegmentAndPose_ns;

  // Create a point cloud of segmentation poses which fall within a time window
  // for the track associated to the segment.
  PointCloud pose_cloud;
  std::vector<Time> pose_times;
  for (const auto& pose: segmentation_poses_.at(segment.track_id)) {
    if (pose.first >= min_time_ns && pose.first <= max_time_ns) {
      pose_cloud.points.push_back(se3ToPclPoint(pose.second));
      pose_times.push_back(pose.first);
    }
  }
  pose_cloud.width = 1;
  pose_cloud.height = pose_cloud.points.size();
  CHECK_GT(pose_times.size(), 0u);

  // Find the nearest pose to the segment within that window.
  pcl::KdTreeFLANN<PclPoint> kd_tree;
  PointCloudPtr pose_cloud_ptr(new PointCloud);
  pcl::copyPointCloud(pose_cloud, *pose_cloud_ptr);
  kd_tree.setInputCloud(pose_cloud_ptr);

  const unsigned int n_nearest_segments = 1u;
  std::vector<int> nearest_neighbour_indices(n_nearest_segments);
  std::vector<float> nearest_neighbour_squared_distances(n_nearest_segments);
  if (kd_tree.nearestKSearch(segment.centroid, n_nearest_segments, nearest_neighbour_indices,
                             nearest_neighbour_squared_distances) <= 0) {
    LOG(ERROR) << "Nearest neighbour search failed.";
  }

  // Return the time of the closest pose.
  return pose_times.at(nearest_neighbour_indices.at(0));
}

void SegMatch::alignTargetMap() {
  segmented_target_cloud_.transform(last_transformation_.inverse());
}

} // namespace segmatch
