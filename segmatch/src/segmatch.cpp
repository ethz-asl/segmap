#include "segmatch/segmatch.hpp"

#include <algorithm>
#include <limits>

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
                                          unsigned int track_id) {
  Clock clock;
  // Save the segmentation pose.
  segmentation_poses_[track_id][latest_pose.time_ns] = latest_pose.T_w;
  last_processed_source_cloud_ = track_id;

  // Apply a cylindrical filter on the input cloud.
  PointICloud filtered_cloud = source_cloud;
  applyCylindricalFilter(laserSlamPoseToPclPoint(latest_pose),
                         params_.segmentation_radius_m,
                         params_.segmentation_height_above_m,
                         params_.segmentation_height_below_m, &filtered_cloud);

  n_points_in_source_.push_back(filtered_cloud.size());

  // Segment the cloud and set segment information.
  if (segmented_source_clouds_.find(track_id) == segmented_source_clouds_.end()) {
    segmented_source_clouds_[track_id] = SegmentedCloud();
  }
  segmenter_->segment(filtered_cloud, &segmented_source_clouds_[track_id]);

  // Filter the boundary segments.
  if (params_.filter_boundary_segments) {
    filterBoundarySegmentsOfSourceCloud(laserSlamPoseToPclPoint(latest_pose),
                                        track_id);
  }

  LOG(INFO) << "Removing too near segments from source map.";
  filterNearestSegmentsInCloud(&segmented_source_clouds_[track_id],
                               params_.centroid_distance_threshold_m, 5u);

  segmented_source_clouds_[track_id].setTimeStampOfSegments(latest_pose.time_ns);
  segmented_source_clouds_[track_id].setLinkPoseOfSegments(latest_pose.T_w);
  segmented_source_clouds_[track_id].setTrackId(track_id);

  // Describe the cloud.
  descriptors_->describe(&segmented_source_clouds_[track_id]);
  clock.takeTime();
  segmentation_and_description_timings_.emplace(latest_pose.time_ns, clock.getRealTime());

  n_segments_in_source_.push_back(segmented_source_clouds_[track_id].getNumberOfValidSegments());
}

void SegMatch::processAndSetAsTargetCloud(const PointICloud& target_cloud) {
  // Process the cloud.
  processCloud(target_cloud, &segmented_target_cloud_);

  // Overwrite the old target.
  classifier_->setTarget(segmented_target_cloud_);
}

void SegMatch::transferSourceToTarget(unsigned int track_id,
                                      laser_slam::Time timestamp_ns) {
  Clock clock;
  target_queue_.push_back(segmented_source_clouds_[track_id]);

  // Remove empty clouds from queue.
  std::vector<SegmentedCloud>::iterator it = target_queue_.begin();
  while(it != target_queue_.end()) {
    if(it->empty()) {
      it = target_queue_.erase(it);
    } else {
      ++it;
    }
  }

  unsigned int num_cloud_transfered = 0u;
  if (!target_queue_.empty()) {
    if (params_.filter_duplicate_segments) {
      LOG(INFO) << "Source cloud size before duplicates: " <<
          target_queue_.at(0u).getNumberOfValidSegments();
      filterDuplicateSegmentsOfTargetMap(&target_queue_.at(0u));
      LOG(INFO) << "Source cloud size after duplicates: " <<
          target_queue_.at(0u).getNumberOfValidSegments();
    }
    segmented_target_cloud_.addSegmentedCloud(target_queue_.at(0u));
    target_queue_.erase(target_queue_.begin());
    ++num_cloud_transfered;
  }

  if (num_cloud_transfered > 0u) {
    classifier_->setTarget(segmented_target_cloud_);
  }
  clock.takeTime();
  source_to_target_timings_.emplace(timestamp_ns, clock.getRealTime());
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

  LOG(INFO) << "Removing too near segments from source map.";
  filterNearestSegmentsInCloud(segmented_cloud, params_.centroid_distance_threshold_m, 5u);

  std::vector<double> segmentation_timings;
  descriptors_->describe(segmented_cloud, &segmentation_timings);
  if (timings != NULL && !segmentation_timings.empty()) {
    // Following timings are description.
    for (size_t i = 0u; i < segmentation_timings.size(); ++i) {
      timings->push_back(segmentation_timings[i]);
    }
  }
}

PairwiseMatches SegMatch::findMatches(PairwiseMatches* matches_after_first_stage,
                                      unsigned int track_id,
                                      laser_slam::Time timestamp_ns) {
  Clock clock;
  PairwiseMatches candidates;
  if (!segmented_source_clouds_[track_id].empty()) {
    candidates = classifier_->findCandidates(segmented_source_clouds_[track_id],
                                             matches_after_first_stage);
  }
  clock.takeTime();
  matching_timings_.emplace(timestamp_ns, clock.getRealTime());
  return candidates;
}

Time findTimeOfClosestPose(const Trajectory& poses,
                           std::vector<Segment>& segments) {
  CHECK(!poses.empty());
  CHECK(!segments.empty());

  // Compute center of segments.
  PclPoint segments_center;
  for (const auto& segment: segments) {
    segments_center.x += segment.centroid.x;
    segments_center.y += segment.centroid.y;
    segments_center.z += segment.centroid.z;
  }
  segments_center.x /= double(segments.size());
  segments_center.y /= double(segments.size());
  segments_center.z /= double(segments.size());

  double minimum_distance_m = std::numeric_limits<double>::max();
  Time closest_pose_time_ns;
  for (const auto& pose: poses) {
    double distance_m = pointToPointDistance(se3ToPclPoint(pose.second), segments_center);
    if (distance_m < minimum_distance_m) {
      minimum_distance_m = distance_m;
      closest_pose_time_ns = pose.first;
    }
  }

  return closest_pose_time_ns;
}

bool SegMatch::filterMatches(const PairwiseMatches& predicted_matches,
                             PairwiseMatches* filtered_matches_ptr,
                             RelativePose* loop_closure,
                             std::vector<PointICloudPair>* matched_segment_clouds,
                             unsigned int track_id,
                             laser_slam::Time timestamp_ns) {
  if (matched_segment_clouds != NULL) { matched_segment_clouds->clear(); }

  PairwiseMatches filtered_matches;
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

  Clock clock;

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

      // Filter the matches by segment timestamps.
      LOG(INFO) << "Filtering the matches based on timestamps.";
      Correspondences time_filtered_clustered_corrs;
      for (const auto& cluster: clustered_corrs) {
        pcl::Correspondences time_filtered_cluster;
        for (const auto& match: cluster) {
          PairwiseMatch pairwise_match = predicted_matches.at(match.index_query);
          Segment source_segment, target_segment;
          if (segmented_source_clouds_[track_id].findValidSegmentById(pairwise_match.ids_.first,
                                                                      &source_segment)) {
            if (segmented_target_cloud_.findValidSegmentById(pairwise_match.ids_.second,
                                                             &target_segment)) {
              if (source_segment.track_id != target_segment.track_id ||
                  std::max(source_segment.timestamp_ns, target_segment.timestamp_ns) >=
                  std::min(source_segment.timestamp_ns, target_segment.timestamp_ns) +
                  params_.min_time_between_segment_for_matches_ns) {
                time_filtered_cluster.push_back(match);
              } else {
                LOG(INFO) << "Removed match with source_segment.timestamp_ns " << source_segment.timestamp_ns
                    << " and target_segment.timestamp_ns " << target_segment.timestamp_ns;
              }
            } else {
              LOG(INFO) << "Could not find target segment when filtering on timestamps";
            }
          } else {
            LOG(INFO) << "Could not find source segment when filtering on timestamps";
          }
        }
        time_filtered_clustered_corrs.push_back(time_filtered_cluster);
        LOG(INFO) << "Cluster had size " << cluster.size() << " before and " <<
            time_filtered_cluster.size() << " after.";
      }
      clustered_corrs = time_filtered_clustered_corrs;

      if (!clustered_corrs.empty()) {
        // Find largest cluster.
        size_t largest_cluster_size = 0;
        size_t largest_cluster_index = 0;
        for (size_t i = 0u; i < clustered_corrs.size(); ++i) {
          LOG(INFO) << "Cluster " << i << " has " << clustered_corrs[i].size() << "segments.";
          if (clustered_corrs[i].size() >= largest_cluster_size) {
            largest_cluster_size = clustered_corrs[i].size();
            largest_cluster_index = i;
          }
        }
        LOG(INFO) << "Largest cluster: " << largest_cluster_size << " matches.";

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

    clock.takeTime();
    geometric_verification_timings_.emplace(timestamp_ns, clock.getRealTime());

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
        segmented_source_clouds_[track_id].findValidSegmentById(filtered_matches[i].ids_.first, &segment);

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
      LOG(INFO) << "Found a loop";

      loops_timestamps_.push_back(timestamp_ns);
      laser_slam::Clock clock;
      // Find the trajectory poses to be linked by the loop-closure.
      // For each segment, find the timestamp of the closest segmentation pose.
      std::vector<Time> source_segmentation_times;
      std::vector<Time> target_segmentation_times;
      std::vector<Id> source_track_ids;
      std::vector<Id> target_track_ids;
      std::vector<Segment> source_segments;
      std::vector<Segment> target_segments;
      for (const auto& match: filtered_matches) {
        Segment segment;
        CHECK(segmented_source_clouds_[track_id].findValidSegmentById(match.ids_.first, &segment));
        source_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
        source_segments.push_back(segment);
        source_track_ids.push_back(segment.track_id);
        LOG(INFO) << "Source Track id " << segment.track_id << " Source segment ID " << segment.segment_id;

        CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
        target_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
        target_segments.push_back(segment);
        target_track_ids.push_back(segment.track_id);
        LOG(INFO) << "Target Track id " << segment.track_id << " Source segment ID " << segment.segment_id;
      }

      const Id source_track_id = findMostOccuringId(source_track_ids);
      const Id target_track_id = findMostOccuringId(target_track_ids);
      LOG(INFO) << "source_track_id " << source_track_id << " target_track_id " <<
          target_track_id;

      const Time target_most_occuring_time = findMostOccuringTime(target_segmentation_times);

      LOG(INFO) << "Finding source_track_time_ns and target_track_time_ns";
      Time source_track_time_ns, target_track_time_ns;
      if (source_track_id != target_track_id) {
        // Get the head of the source trajectory.
        Time trajectory_last_time_ns = segmentation_poses_[source_track_id].rbegin()->first;
        Time start_time_of_head_ns;

        if (trajectory_last_time_ns > params_.min_time_between_segment_for_matches_ns) {
          start_time_of_head_ns = trajectory_last_time_ns - params_.min_time_between_segment_for_matches_ns;
        } else {
          start_time_of_head_ns = 0u;
        }

        LOG(INFO) << "start_time_of_head_ns " << start_time_of_head_ns;

        Trajectory head_poses;

        for (const auto pose: segmentation_poses_[source_track_id]) {
          if (pose.first > start_time_of_head_ns) {
            head_poses.emplace(pose.first, pose.second);
          }
        }

        LOG(INFO) << "head_poses.size() " << head_poses.size();

        // Get a window over the target trajectory.
        const Time half_window_size_ns = 180000000000u;
        const Time window_max_value_ns = target_most_occuring_time + half_window_size_ns;
        Time window_min_value_ns;
        if (target_most_occuring_time > half_window_size_ns) {
          window_min_value_ns = target_most_occuring_time - half_window_size_ns;
        } else {
          window_min_value_ns = 0u;
        }
        Trajectory poses_in_window;
        for (const auto& pose: segmentation_poses_[target_track_id]) {
          if (pose.first >= window_min_value_ns &&
              pose.first <=  window_max_value_ns) {

            // Compute center of segments.
            PclPoint segments_center;
            for (const auto& segment: target_segments) {
              segments_center.z += segment.centroid.z;
            }
            segments_center.z /= double(target_segments.size());

            // Check that pose lies below the segments center of mass.
            if (!params_.check_pose_lies_below_segments ||
                pose.second.getPosition()(2) < segments_center.z) {
              poses_in_window.emplace(pose.first, pose.second);
            }
          }
        }

        source_track_time_ns =  findTimeOfClosestPose(head_poses,
                                                      source_segments);
        target_track_time_ns =  findTimeOfClosestPose(poses_in_window,
                                                      target_segments);
      } else {
        // Split the trajectory into head and tail.
        Time trajectory_last_time_ns = segmentation_poses_[source_track_id].rbegin()->first;
        CHECK_GT(trajectory_last_time_ns, params_.min_time_between_segment_for_matches_ns);
        Time start_time_of_head_ns = trajectory_last_time_ns -
            params_.min_time_between_segment_for_matches_ns;

        Trajectory tail_poses, head_poses;

        for (const auto pose: segmentation_poses_[source_track_id]) {
          if (pose.first < start_time_of_head_ns) {
            tail_poses.emplace(pose.first, pose.second);
          } else {
            head_poses.emplace(pose.first, pose.second);
          }
        }

        source_track_time_ns =  findTimeOfClosestPose(head_poses,
                                                      source_segments);
        target_track_time_ns =  findTimeOfClosestPose(tail_poses,
                                                      target_segments);
      }

      LOG(INFO) << "Took " <<
          clock.getRealTime() << " ms to find the LC pose times.";

      LOG(INFO) << "source_track_time_ns " << source_track_time_ns;
      LOG(INFO) << "target_track_time_ns " << target_track_time_ns;

      loop_closure->time_a_ns = target_track_time_ns;
      loop_closure->time_b_ns = source_track_time_ns;
      loop_closure->track_id_a = target_track_id;
      loop_closure->track_id_b = source_track_id;

      SE3 w_T_a_b = fromApproximateTransformationMatrix(transformation);
      loop_closure->T_a_b = w_T_a_b;

      // Save the loop closure.
      loop_closures_.push_back(*loop_closure);
    }

    // Save a copy of the fitered matches.
    last_filtered_matches_ = filtered_matches;
    last_predicted_matches_ = predicted_matches;
  }

  return !filtered_matches.empty();
}

void SegMatch::update(const std::vector<laser_slam::Trajectory>& trajectories) {
  Clock clock;
  CHECK_EQ(trajectories.size(), segmentation_poses_.size());
  // Update the segmentation positions.
  for (size_t i = 0u; i < trajectories.size(); ++i) {
    for (auto& pose: segmentation_poses_[i]){
      pose.second = trajectories.at(i).at(pose.first);
    }
  }
  // Update the source, target and clouds in the buffer.
  for (auto& source_cloud: segmented_source_clouds_) {
    source_cloud.second.updateSegments(trajectories);
  }
  segmented_target_cloud_.updateSegments(trajectories);
  for (auto& segmented_cloud: target_queue_) {
    segmented_cloud.updateSegments(trajectories);
  }

  // Update the last filtered matches.
  for (auto& match: last_filtered_matches_) {
    Segment segment;
    // TODO Replaced the CHECK with a if. How should we handle the case
    // when one segment was removed during duplicate check?
    if (segmented_source_clouds_[last_processed_source_cloud_].
        findValidSegmentById(match.ids_.first, &segment)) {
      match.centroids_.first = segment.centroid;
    }

    if (segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment)) {
      match.centroids_.second = segment.centroid;
    }
  }

  for (auto& match: last_predicted_matches_) {
    Segment segment;
    if (segmented_source_clouds_[last_processed_source_cloud_].
        findValidSegmentById(match.ids_.first, &segment)) {
      match.centroids_.first = segment.centroid;
    }

    if (segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment)) {
      match.centroids_.second = segment.centroid;
    }
  }

  // Filter duplicates.
  LOG(INFO) << "Removing too near segments from target map.";
  filterNearestSegmentsInCloud(&segmented_target_cloud_, params_.centroid_distance_threshold_m,
                               5u);

  clock.takeTime();
  update_timings_.emplace(trajectories[0u].rbegin()->first, clock.getRealTime());
}

void SegMatch::getSourceRepresentation(PointICloud* source_representation,
                                       const double& distance_to_raise,
                                       unsigned int track_id) const {
  if (segmented_source_clouds_.find(track_id) != segmented_source_clouds_.end()) {
    segmentedCloudToCloud(segmented_source_clouds_.at(track_id).transformed(
        Eigen::Affine3f(Eigen::Translation3f(0,0,distance_to_raise)).matrix()),
        source_representation);
  }
}

void SegMatch::getTargetRepresentation(PointICloud* target_representation) const {
  segmentedCloudToCloud(segmented_target_cloud_
                        , target_representation);
}

void SegMatch::getTargetSegmentsCentroids(PointICloud* segments_centroids) const {
  CHECK_NOTNULL(segments_centroids);
  PointICloud cloud;
  std::vector<int> permuted_indexes;
  for (unsigned int i = 0u; i < segmented_target_cloud_.getNumberOfValidSegments(); ++i) {
    permuted_indexes.push_back(i);
  }
  std::random_shuffle(permuted_indexes.begin(), permuted_indexes.end());
  unsigned int i = 0u;
  for (std::unordered_map<Id, Segment>::const_iterator it = segmented_target_cloud_.begin();
      it != segmented_target_cloud_.end(); ++it) {
    PointI centroid;
    Segment segment = it->second;
    centroid.x = segment.centroid.x;
    centroid.y = segment.centroid.y;
    centroid.z = segment.centroid.z;
    centroid.intensity = permuted_indexes[i];
    cloud.points.push_back(centroid);
    ++i;
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();
  // TODO use move to to avoid deep copy.
  *segments_centroids = cloud;
}

void SegMatch::getSourceSegmentsCentroids(PointICloud* segments_centroids,
                                          unsigned int track_id) const {
  // TODO combine with function above and reuse code.
  CHECK_NOTNULL(segments_centroids);
  if (segmented_source_clouds_.find(track_id) != segmented_source_clouds_.end()) {
    PointICloud cloud;
    std::vector<int> permuted_indexes;
    for (unsigned int i = 0u; i < segmented_source_clouds_.at(track_id).getNumberOfValidSegments(); ++i) {
      permuted_indexes.push_back(i);
    }
    std::random_shuffle(permuted_indexes.begin(), permuted_indexes.end());
    unsigned int i = 0u;
    for (std::unordered_map<Id, Segment>::const_iterator it =
        segmented_source_clouds_.at(track_id).begin();it !=
            segmented_source_clouds_.at(track_id).end(); ++it) {
      PointI centroid;
      Segment segment = it->second;
      centroid.x = segment.centroid.x;
      centroid.y = segment.centroid.y;
      centroid.z = segment.centroid.z;
      centroid.intensity = permuted_indexes[i];
      cloud.points.push_back(centroid);
      ++i;
    }
    cloud.width = 1;
    cloud.height = cloud.points.size();
    // TODO use move to to avoid deep copy.
    *segments_centroids = cloud;
  }
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

void SegMatch::filterBoundarySegmentsOfSourceCloud(const PclPoint& center,
                                                   unsigned int track_id) {
  if (!segmented_source_clouds_[track_id].empty()) {
    const double squared_radius_m = (params_.segmentation_radius_m - 1.0) *
        (params_.segmentation_radius_m -1.0);

    const double height_above_m = params_.segmentation_height_above_m - 1.0;
    const double height_below_m = params_.segmentation_height_below_m - 1.0;

    // Get a list of segments with at least one point outside the boundary.
    std::vector<Id> boundary_segments_ids;
    for (std::unordered_map<Id, Segment>::const_iterator it =
        segmented_source_clouds_[track_id].begin(); it != segmented_source_clouds_[track_id].end();
        ++it) {
      Segment segment = it->second;
      PointICloud segment_cloud = segment.point_cloud;
      // Loop over points until one is found outside of the boundary.
      for (size_t j = 0u; j < segment_cloud.size(); ++j) {
        PointI point = segment_cloud.at(j);
        point.x -= center.x;
        point.y -= center.y;
        if ((point.x * point.x + point.y * point.y) >= squared_radius_m) {
          // If found, add the segment to the deletion list, and move on to the next segment.
          boundary_segments_ids.push_back(segment.segment_id);
          break;
        }
        if (point.z > center.z + height_above_m) {
          boundary_segments_ids.push_back(segment.segment_id);
          break;
        }
        if (point.z < center.z - height_below_m){
          boundary_segments_ids.push_back(segment.segment_id);
          break;
        }
      }
    }

    // Remove boundary segments.
    size_t n_removals;
    segmented_source_clouds_[track_id].deleteSegmentsById(boundary_segments_ids, &n_removals);
    LOG(INFO) << "Removed " << n_removals << " boundary segments.";
  }
}

void SegMatch::filterDuplicateSegmentsOfTargetMap(SegmentedCloud* cloud_to_be_added) {
  if (!cloud_to_be_added->empty()) {
    laser_slam::Clock clock;
    std::vector<Id> duplicate_segments_ids;
    std::vector<Id> duplicate_segments_ids_in_cloud_to_add;
    std::vector<Id> target_segment_ids;

    // Get a cloud with segments centroids which are close to the cloud to be added.
    PointCloud centroid_cloud = segmented_target_cloud_.centroidsAsPointCloud(
        cloud_to_be_added->begin()->second.T_w_linkpose,
        params_.segmentation_radius_m * 3.0,
        &target_segment_ids);

    const double minimum_distance_squared = params_.centroid_distance_threshold_m *
        params_.centroid_distance_threshold_m;

    unsigned int n_nearest_segments = 4u;
    const laser_slam::Time max_time_diff_ns = 60000000000u;
    if (!target_segment_ids.empty()) {
      n_nearest_segments = std::min(static_cast<unsigned int>(target_segment_ids.size()), n_nearest_segments);
      // Set up nearest neighbour search.
      pcl::KdTreeFLANN<PclPoint> kdtree;
      PointCloudPtr centroid_cloud_ptr(new PointCloud);
      pcl::copyPointCloud(centroid_cloud, *centroid_cloud_ptr);
      kdtree.setInputCloud(centroid_cloud_ptr);

      for (std::unordered_map<Id, Segment>::const_iterator it = cloud_to_be_added->begin();
          it != cloud_to_be_added->end(); ++it) {
        std::vector<int> nearest_neighbour_indice(n_nearest_segments);
        std::vector<float> nearest_neighbour_squared_distance(n_nearest_segments);

        // Find the nearest neighbours.
        if (kdtree.nearestKSearch(it->second.centroid,
                                  n_nearest_segments, nearest_neighbour_indice,
                                  nearest_neighbour_squared_distance) <= 0) {
          LOG(ERROR) << "Nearest neighbour search failed.";
        }

        for (size_t i = 0u; i < n_nearest_segments; ++i) {
          // Check if within distance.
          if (nearest_neighbour_squared_distance[i] <= minimum_distance_squared) {
            Segment other_segment;
            segmented_target_cloud_.findValidSegmentById(
                target_segment_ids[nearest_neighbour_indice[i]], &other_segment);
            if (other_segment.track_id == it->second.track_id) {
              // If inside the window, remove the old one. Otherwise remove current one.
              if (it->second.timestamp_ns < other_segment.timestamp_ns + max_time_diff_ns) {
                if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                              other_segment.segment_id) ==
                                  duplicate_segments_ids.end()) {
                  duplicate_segments_ids.push_back(other_segment.segment_id);
                }
              } else {
                if (std::find(duplicate_segments_ids_in_cloud_to_add.begin(),
                              duplicate_segments_ids_in_cloud_to_add.end(),
                              it->second.segment_id) ==
                                  duplicate_segments_ids_in_cloud_to_add.end()) {
                  duplicate_segments_ids_in_cloud_to_add.push_back(it->second.segment_id);
                }
              }
            } else {
              // Remove current segment if other is from another trajectory.
              if (std::find(duplicate_segments_ids_in_cloud_to_add.begin(),
                            duplicate_segments_ids_in_cloud_to_add.end(),
                            it->second.segment_id) ==
                                duplicate_segments_ids_in_cloud_to_add.end()) {
                duplicate_segments_ids_in_cloud_to_add.push_back(it->second.segment_id);
              }
            }
          }
        }
      }
    }

    // Remove duplicates.
    size_t n_removals;
    segmented_target_cloud_.deleteSegmentsById(duplicate_segments_ids, &n_removals);
    clock.takeTime();
    LOG(INFO) << "Removed " << n_removals << " duplicate segments in target map in " <<
        clock.getRealTime() << " ms.";
    cloud_to_be_added->deleteSegmentsById(duplicate_segments_ids_in_cloud_to_add, &n_removals);
    LOG(INFO) << "Removed " << n_removals << " duplicate segments in source map.";
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

  // Overwrite the old target.
  classifier_->setTarget(segmented_target_cloud_);

  // Update the last filtered matches.
  for (auto& match: last_filtered_matches_) {
    Segment segment;
    CHECK(segmented_source_cloud_.findValidSegmentById(match.ids_.first, &segment));
    match.centroids_.first = segment.centroid;
    CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
    match.centroids_.second = segment.centroid;
  }
}

void SegMatch::filterNearestSegmentsInCloud(SegmentedCloud* cloud, double minimum_distance_m,
                                            unsigned int n_nearest_segments) {
  laser_slam::Clock clock;
  std::vector<Id> duplicate_segments_ids;
  std::vector<Id> segment_ids;

  const double minimum_distance_squared = minimum_distance_m * minimum_distance_m;

  // Get a cloud with segments centroids.
  PointCloud centroid_cloud = cloud->centroidsAsPointCloud(&segment_ids);

  LOG(INFO) << "segment_ids.size() " << segment_ids.size();

  if (segment_ids.size() > 2u) {
    n_nearest_segments = std::min(static_cast<unsigned int>(segment_ids.size()), n_nearest_segments);
    // Set up nearest neighbour search.
    pcl::KdTreeFLANN<PclPoint> kdtree;
    PointCloudPtr centroid_cloud_ptr(new PointCloud);
    pcl::copyPointCloud(centroid_cloud, *centroid_cloud_ptr);
    kdtree.setInputCloud(centroid_cloud_ptr);

    for (std::unordered_map<Id, Segment>::const_iterator it = cloud->begin();
        it != cloud->end(); ++it) {

      // If this id is not already in the list to be removed.
      if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                    it->second.segment_id) == duplicate_segments_ids.end()) {

        std::vector<int> nearest_neighbour_indice(n_nearest_segments);
        std::vector<float> nearest_neighbour_squared_distance(n_nearest_segments);

        // Find the nearest neighbours.
        if (kdtree.nearestKSearch(it->second.centroid,
                                  n_nearest_segments, nearest_neighbour_indice,
                                  nearest_neighbour_squared_distance) <= 0) {
          LOG(ERROR) << "Nearest neighbour search failed.";
        }

        for (unsigned int i = 1u; i < n_nearest_segments; ++i) {
          // Check if within distance.
          if (nearest_neighbour_squared_distance[i] <= minimum_distance_squared) {
            Segment other_segment;
            cloud->findValidSegmentById(
                segment_ids[nearest_neighbour_indice[i]], &other_segment);

            // Keep the oldest segment.
            Id id_to_remove;
            if (it->second.timestamp_ns != other_segment.timestamp_ns) {
              if (it->second.timestamp_ns > other_segment.timestamp_ns) {
                id_to_remove = it->second.segment_id;
                // Add id to remove if not already in the list.
                if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                              id_to_remove) == duplicate_segments_ids.end()) {
                  duplicate_segments_ids.push_back(id_to_remove);
                }
                break;
              } else {
                id_to_remove = other_segment.segment_id;
              }
            } else if (it->second.point_cloud.size() >  other_segment.point_cloud.size()) {
              id_to_remove = other_segment.segment_id;
            } else {
              id_to_remove = it->second.segment_id;
              // Add id to remove if not already in the list.
              if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                            id_to_remove) == duplicate_segments_ids.end()) {
                duplicate_segments_ids.push_back(id_to_remove);
              }
              break;
            }

            // Add id to remove if not already in the list.
            if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                          id_to_remove) == duplicate_segments_ids.end()) {
              duplicate_segments_ids.push_back(id_to_remove);
            }
          }
        }
      }
    }
  }

  // Remove duplicates.
  size_t n_removals;
  cloud->deleteSegmentsById(duplicate_segments_ids, &n_removals);
  clock.takeTime();
  LOG(INFO) << "Removed " << n_removals << " duplicate segments in " <<
      clock.getRealTime() << " ms.";
}

void SegMatch::displayTimings() const {

}

void SegMatch::saveTimings() const {
  Eigen::MatrixXd matrix;
  toEigenMatrixXd(segmentation_and_description_timings_, &matrix);
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_segmentation_and_description.csv");

  toEigenMatrixXd(matching_timings_, &matrix);
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_matching.csv");

  toEigenMatrixXd(geometric_verification_timings_, &matrix);
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_geometric_verification.csv");

  toEigenMatrixXd(source_to_target_timings_, &matrix);
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_source_to_target_timings.csv");

  toEigenMatrixXd(update_timings_, &matrix);
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_updates.csv");

  matrix.resize(loops_timestamps_.size(), 1);
  for (size_t i = 0u; i < loops_timestamps_.size(); ++i) {
    matrix(i,0) = loops_timestamps_[i];
  }
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_loops.csv");
}

} // namespace segmatch
