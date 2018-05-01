#include "segmatch/segmatch.hpp"

#include <algorithm>
#include <limits>

#include <laser_slam/benchmarker.hpp>
#include <laser_slam/common.hpp>

#include "segmatch/points_neighbors_providers/kdtree_points_neighbors_provider.hpp"
#include "segmatch/recognizers/correspondence_recognizer_factory.hpp"
#include "segmatch/segmenters/segmenter_factory.hpp"
#include "segmatch/rviz_utilities.hpp"

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
  recognizers_.clear();
}

void SegMatch::init(const SegMatchParams& params,
                    unsigned int num_tracks) {
  params_ = params;

  CorrespondenceRecognizerFactory recognizer_factory(params);
  SegmenterFactory segmenter_factory(params);

  descriptors_ = std::unique_ptr<Descriptors>(new Descriptors(params.descriptors_params));
  segmenter_ = segmenter_factory.create();
  classifier_ = std::unique_ptr<OpenCvRandomForest>(
      new OpenCvRandomForest(params.classifier_params));


  // Create containers for the segmentation poses and recognizers
  CHECK_GT(num_tracks, 0u);
  for (unsigned int i = 0u; i < num_tracks; ++i) {
    recognizers_.emplace_back(recognizer_factory.create());
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

void SegMatch::processAndSetAsSourceCloud(
    LocalMapT& local_map,
    const laser_slam::Pose& latest_pose,
    unsigned int track_id) {
  // Save the segmentation pose.
  segmentation_poses_[track_id][latest_pose.time_ns] = latest_pose.T_w;
  last_processed_source_cloud_ = track_id;

  // Segment the cloud and set segment information.
  if (segmented_source_clouds_.find(track_id) == segmented_source_clouds_.end()) {
    segmented_source_clouds_[track_id] = SegmentedCloud();
  }

  // TODO: It would be better to pass the local map to the segmenter instead of all arguments
  // separately. Even better: the segmenter should live inside the local map (like the normal
  // estimator). Similarly, a class describing the target map would also contain its own segmenter.
  // The second option would make it easier to hide incremental logic and variables behind the
  // local map.
  segmenter_->segment(local_map.getNormals(), local_map.getIsNormalModifiedSinceLastUpdate(),
                      local_map.getFilteredPoints(), local_map.getPointsNeighborsProvider(),
                      segmented_source_clouds_[track_id], local_map.getClusterToSegmentIdMapping(),
                      renamed_segments_[track_id]);
  BENCHMARK_RECORD_VALUE("SM.NumValidSegments", segmented_source_clouds_[track_id].size());

  for (const auto& merge_event : renamed_segments_[track_id]) {
    merge_events_.push_back(database::MergeEvent(latest_pose.time_ns, merge_event.first,
                                                 merge_event.second));
  }

  segmented_source_clouds_[track_id].setTimeStampOfSegments(latest_pose.time_ns);
  segmented_source_clouds_[track_id].setLinkPoseOfSegments(latest_pose.T_w);
  segmented_source_clouds_[track_id].setTrackId(track_id);

  // Describe the cloud.
  BENCHMARK_START("SM.Worker.Describe");
  descriptors_->describe(&segmented_source_clouds_[track_id]);
  BENCHMARK_STOP("SM.Worker.Describe");
  BENCHMARK_RECORD_VALUE("SM.TargetMapSegments", segmented_target_cloud_.size());
}

void SegMatch::processAndSetAsTargetCloud(MapCloud& target_cloud) {
  // Process the cloud.
  processCloud(target_cloud, &segmented_target_cloud_);

  // Overwrite the old target.
  classifier_->setTarget(segmented_target_cloud_);
}

void SegMatch::transferSourceToTarget(unsigned int track_id,
                                      laser_slam::Time timestamp_ns) {
  BENCHMARK_BLOCK("SM.Worker.transferSourceToTarget");
  segmented_target_cloud_.addSegmentedCloud(segmented_source_clouds_[track_id],
                                            renamed_segments_[track_id]);

  filterNearestSegmentsInCloud(segmented_target_cloud_, params_.centroid_distance_threshold_m,
                               5u);

  classifier_->setTarget(segmented_target_cloud_);
}

void SegMatch::processCloud(MapCloud& cloud,
                            SegmentedCloud* segmented_cloud) {
  // Build a kd-tree of the cloud.
  MapCloud::ConstPtr cloud_ptr(&cloud, [](MapCloud const* ptr) {});
  KdTreePointsNeighborsProvider<MapPoint> kd_tree;
  kd_tree.update(cloud_ptr);

  // Estimate normals if necessary.
  PointNormals empty_normals;
  PointNormals const * normals = &empty_normals;
  std::unique_ptr<NormalEstimator> normal_estimator;
  if (params_.segmenter_params.segmenter_type == "SimpleSmoothnessConstraints" ||
      params_.segmenter_params.segmenter_type == "IncrementalSmoothnessConstraints") {
    // Create normal estimator
    normal_estimator = NormalEstimator::create(
        params_.normal_estimator_type, params_.radius_for_normal_estimation_m);

    // Estimate normals
    // All points are considered new points.
    std::vector<int> new_points_indices(cloud.size());
    std::iota(new_points_indices.begin(), new_points_indices.end(), 0u);
    normal_estimator->updateNormals(cloud, { }, new_points_indices, kd_tree);
    normals = &normal_estimator->getNormals();
  }

  std::vector<Id> segment_ids;
  std::vector<std::pair<Id, Id>> renamed_segments;
  std::vector<bool> is_point_modified(cloud.size(), false);
  segmenter_->segment(*normals, is_point_modified, cloud, kd_tree, *segmented_cloud, segment_ids,
                      renamed_segments);

  LOG(INFO) << "Removing too near segments from source map.";
  filterNearestSegmentsInCloud(*segmented_cloud, params_.centroid_distance_threshold_m, 5u);

  descriptors_->describe(segmented_cloud);
}

PairwiseMatches SegMatch::findMatches(PairwiseMatches* matches_after_first_stage,
                                      unsigned int track_id,
                                      laser_slam::Time timestamp_ns) {
  BENCHMARK_BLOCK("SM.Worker.FindMatches");
  PairwiseMatches candidates;
  if (!segmented_source_clouds_[track_id].empty()) {
    candidates = classifier_->findCandidates(segmented_source_clouds_[track_id],
                                             matches_after_first_stage);
  }
  return candidates;
}

Time findTimeOfClosestPose(const Trajectory& poses,
                           std::vector<Segment>& segments) {
  CHECK(!poses.empty());
  CHECK(!segments.empty());

  // Compute center of segments.
  PclPoint segments_center;
  for (const auto& segment : segments) {
    segments_center.getVector3fMap() += segment.getLastView().centroid.getVector3fMap();
  }
  segments_center.x /= double(segments.size());
  segments_center.y /= double(segments.size());
  segments_center.z /= double(segments.size());

  double minimum_distance_m = std::numeric_limits<double>::max();
  Time closest_pose_time_ns;
  for (const auto& pose : poses) {
    double distance_m = pointToPointDistance(se3ToPclPoint(pose.second), segments_center);
    if (distance_m < minimum_distance_m) {
      minimum_distance_m = distance_m;
      closest_pose_time_ns = pose.first;
    }
  }

  return closest_pose_time_ns;
}

PairwiseMatches SegMatch::filterMatches(const PairwiseMatches& predicted_matches,
                                        const unsigned int track_id) {
  BENCHMARK_BLOCK("SM.Worker.FilterMatches");

  unsigned int n_cars = 0u;
  for (const auto& segment : segmented_target_cloud_) {
    if (segment.second.getLastView().semantic == 1) {
      n_cars++;
    }
  }

  // Save a copy of the predicted matches.
  last_predicted_matches_ = predicted_matches;
  if (predicted_matches.empty()) return PairwiseMatches();

  // Filter the matches by segment timestamps.
  PairwiseMatches filtered_matches;
  for (const auto& pairwise_match: predicted_matches) {
    Segment *source_segment, *target_segment;
    if (!segmented_source_clouds_[track_id].findValidSegmentPtrById(pairwise_match.ids_.first,
                                                                    &source_segment)) {
      LOG(INFO) << "Could not find source segment when filtering on timestamps";
    } else if (!segmented_target_cloud_.findValidSegmentPtrById(pairwise_match.ids_.second,
                                                                &target_segment)) {
      LOG(INFO) << "Could not find target segment when filtering on timestamps";
    } else {
      const Time segments_time_difference_ns =
          std::max(source_segment->getLastView().timestamp_ns,
                   target_segment->getLastView().timestamp_ns) -
                   std::min(source_segment->getLastView().timestamp_ns,
                            target_segment->getLastView().timestamp_ns);

      if (source_segment->track_id != target_segment->track_id ||
          segments_time_difference_ns >= params_.min_time_between_segment_for_matches_ns) {
        filtered_matches.push_back(pairwise_match);
      }
    }
  }

  return filtered_matches;
}

const PairwiseMatches& SegMatch::recognize(const PairwiseMatches& predicted_matches,
                                           const unsigned int track_id,
                                           const laser_slam::Time timestamp_ns,
                                           laser_slam::RelativePose* loop_closure) {

  BENCHMARK_BLOCK("SM.Worker.Recognition");

  std::unique_ptr<CorrespondenceRecognizer>& recognizer = recognizers_[track_id];
  recognizer->recognize(predicted_matches);

  // Assume that the matches in the first cluster are true positives. Return in case recognition
  // was unsuccessful.
  const std::vector<PairwiseMatches>& candidate_clusters =
      recognizer->getCandidateClusters();
  if (candidate_clusters.empty()) {
    last_filtered_matches_ = PairwiseMatches();
    return last_filtered_matches_;
  } else {
    last_filtered_matches_ = candidate_clusters.front();
  }

  // If desired, return the loop-closure.
  if (loop_closure != nullptr) {
    BENCHMARK_BLOCK("SM.Worker.Recognition.GetLoopClosure");
    loops_timestamps_.push_back(timestamp_ns);

    // Find the trajectory poses to be linked by the loop-closure.
    // For each segment, find the timestamp of the closest segmentation pose.
    std::vector<Time> source_segmentation_times;
    std::vector<Time> target_segmentation_times;
    std::vector<Id> source_track_ids;
    std::vector<Id> target_track_ids;
    std::vector<Segment> source_segments;
    std::vector<Segment> target_segments;
    for (const auto& match : last_filtered_matches_) {
      Segment segment;
      CHECK(segmented_source_clouds_[track_id].findValidSegmentById(match.ids_.first, &segment));
      source_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
      source_segments.push_back(segment);
      source_track_ids.push_back(segment.track_id);

      CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
      target_segmentation_times.push_back(findTimeOfClosestSegmentationPose(segment));
      target_segments.push_back(segment);
      target_track_ids.push_back(segment.track_id);
    }

    const Id source_track_id = findMostOccuringElement(source_track_ids);
    const Id target_track_id = findMostOccuringElement(target_track_ids);
    LOG(INFO) << "Found a loop between source_track_id " << source_track_id << " target_track_id " <<
        target_track_id;

    const Time target_most_occuring_time = findMostOccuringElement(target_segmentation_times);

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

      Trajectory head_poses;

      for (const auto pose: segmentation_poses_[source_track_id]) {
        if (pose.first > start_time_of_head_ns) {
          head_poses.emplace(pose.first, pose.second);
        }
      }

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
            segments_center.z += segment.getLastView().centroid.z;
          }
          segments_center.z /= double(target_segments.size());

          // Check that pose lies below the segments center of mass.
          if (!params_.check_pose_lies_below_segments ||
              pose.second.getPosition()(2) < segments_center.z) {
            poses_in_window.emplace(pose.first, pose.second);
          }
        }
      }

      source_track_time_ns =  findTimeOfClosestPose(head_poses, source_segments);
      target_track_time_ns =  findTimeOfClosestPose(poses_in_window, target_segments);
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

      source_track_time_ns =  findTimeOfClosestPose(head_poses, source_segments);
      target_track_time_ns =  findTimeOfClosestPose(tail_poses, target_segments);
    }

    loop_closure->time_a_ns = target_track_time_ns;
    loop_closure->time_b_ns = source_track_time_ns;
    loop_closure->track_id_a = target_track_id;
    loop_closure->track_id_b = source_track_id;

    // Again, assume that the best transformation is the correct one.
    CHECK_GT(recognizer->getCandidateTransformations().size(), 0u);
    SE3 w_T_a_b = fromApproximateTransformationMatrix(
        recognizer->getCandidateTransformations().front());
    loop_closure->T_a_b = w_T_a_b;

    // Save the loop closure.
    loop_closures_.push_back(*loop_closure);
  }

  return last_filtered_matches_;
}

void SegMatch::update(const std::vector<laser_slam::Trajectory>& trajectories) {
  BENCHMARK_BLOCK("SM.Update");
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
  for (auto& match : last_filtered_matches_) {
    Segment segment;
    // TODO Replaced the CHECK with a if. How should we handle the case
    // when one segment was removed during duplicate check?
    if (segmented_source_clouds_[last_processed_source_cloud_].
        findValidSegmentById(match.ids_.first, &segment)) {
      match.centroids_.first = segment.getLastView().centroid;
    }

    if (segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment)) {
      match.centroids_.second = segment.getLastView().centroid;
    }
  }

  for (auto& match : last_predicted_matches_) {
    Segment segment;
    if (segmented_source_clouds_[last_processed_source_cloud_].
        findValidSegmentById(match.ids_.first, &segment)) {
      match.centroids_.first = segment.getLastView().centroid;
    }

    if (segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment)) {
      match.centroids_.second = segment.getLastView().centroid;
    }
  }

  // Filter duplicates.
  LOG(INFO) << "Removing too near segments from target map.";
  filterNearestSegmentsInCloud(segmented_target_cloud_, params_.centroid_distance_threshold_m,
                               5u);

  classifier_->setTarget(segmented_target_cloud_);
}

void SegMatch::getSourceRepresentation(PointICloud* source_representation,
                                       const double& distance_to_raise,
                                       unsigned int track_id) const {
  if (segmented_source_clouds_.find(track_id) !=
      segmented_source_clouds_.end()) {
    Eigen::Affine3f transform(Eigen::Translation3f(0, 0, distance_to_raise));
    *source_representation = RVizUtilities::segmentedCloudtoPointICloud(
        segmented_source_clouds_.at(track_id).transformed(transform.matrix()));
  }
}

void SegMatch::getSourceSemantics(PointICloud* source_semantics,
                                  const double& distance_to_raise,
                                  unsigned int track_id) const {
  if (segmented_source_clouds_.find(track_id) !=
      segmented_source_clouds_.end()) {
    Eigen::Affine3f transform(Eigen::Translation3f(0, 0, distance_to_raise));
    *source_semantics = RVizUtilities::segmentedCloudSemanticstoPointICloud(
        segmented_source_clouds_.at(track_id).transformed(transform.matrix()));
  }
}

void SegMatch::getTargetRepresentation(PointICloud* target_representation,
                                       bool get_compressed) const {
  *target_representation = RVizUtilities::segmentedCloudtoPointICloud(
      segmented_target_cloud_, get_compressed);
}

void SegMatch::getTargetReconstruction(PointICloud* target_reconstruction,
                                       bool get_compressed) const {
  *target_reconstruction = RVizUtilities::segmentedCloudSemanticstoPointICloud(
      segmented_target_cloud_, true, get_compressed);
}

void SegMatch::getSourceReconstruction(PointICloud* source_reconstruction,
                                       unsigned int track_id) const {
  if (segmented_source_clouds_.find(track_id) !=  segmented_source_clouds_.end()) {
    *source_reconstruction = RVizUtilities::segmentedCloudtoPointICloud(
        segmented_source_clouds_.at(track_id), false, true);
  }
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
    centroid.x = segment.getLastView().centroid.x;
    centroid.y = segment.getLastView().centroid.y;
    centroid.z = segment.getLastView().centroid.z;
    centroid.intensity = permuted_indexes[i];
    cloud.points.push_back(centroid);
    ++i;
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();
  // TODO use move to to avoid deep copy.
  *segments_centroids = cloud;
}

void SegMatch::getTargetSegmentsCentroidsWithTrajIdAsIntensity(PointICloud* segments_centroids) const {
  CHECK_NOTNULL(segments_centroids);
  PointICloud cloud;
  for (std::unordered_map<Id, Segment>::const_iterator it = segmented_target_cloud_.begin();
      it != segmented_target_cloud_.end(); ++it) {
    PointI centroid;
    Segment segment = it->second;
    centroid.x = segment.getLastView().centroid.x;
    centroid.y = segment.getLastView().centroid.y;
    centroid.z = segment.getLastView().centroid.z;
    centroid.intensity = segment.track_id;
    cloud.points.push_back(centroid);
  }
  cloud.width = 1;
  cloud.height = cloud.points.size();

  *segments_centroids = std::move(cloud);
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
      centroid.x = segment.getLastView().centroid.x;
      centroid.y = segment.getLastView().centroid.y;
      centroid.z = segment.getLastView().centroid.z;
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

Time SegMatch::findTimeOfClosestSegmentationPose(const Segment& segment) const {
  const Time segment_time_ns = segment.getLastView().timestamp_ns;

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
  if (kd_tree.nearestKSearch(segment.getLastView().centroid, n_nearest_segments,
                             nearest_neighbour_indices,
                             nearest_neighbour_squared_distances) <= 0) {
    LOG(ERROR) << "Nearest neighbour search failed.";
  }

  // Return the time of the closest pose.
  return pose_times.at(nearest_neighbour_indices.at(0));
}

void SegMatch::alignTargetMap() {
  segmented_source_clouds_[last_processed_source_cloud_].transform(last_transformation_.inverse());

  // Overwrite the old target.
  classifier_->setTarget(segmented_target_cloud_);

  // Update the last filtered matches.
  for (auto& match: last_filtered_matches_) {
    Segment segment;
    CHECK(segmented_source_clouds_.at(last_processed_source_cloud_).
          findValidSegmentById(match.ids_.first, &segment));
    match.centroids_.first = segment.getLastView().centroid;
    CHECK(segmented_target_cloud_.findValidSegmentById(match.ids_.second, &segment));
    match.centroids_.second = segment.getLastView().centroid;
  }
}

void SegMatch::filterNearestSegmentsInCloud(SegmentedCloud& cloud, double minimum_distance_m,
                                            unsigned int n_nearest_segments) {
  std::vector<Id> duplicate_segments_ids;
  std::vector<Id> segment_ids;

  const double minimum_distance_squared = minimum_distance_m * minimum_distance_m;

  // Get a cloud with segments centroids.
  PointCloud centroid_cloud = cloud.centroidsAsPointCloud(segment_ids);

  if (segment_ids.size() > 2u) {
    n_nearest_segments = std::min(static_cast<unsigned int>(segment_ids.size()), n_nearest_segments);
    // Set up nearest neighbour search.
    pcl::KdTreeFLANN<PclPoint> kdtree;
    PointCloudPtr centroid_cloud_ptr(new PointCloud);
    pcl::copyPointCloud(centroid_cloud, *centroid_cloud_ptr);
    kdtree.setInputCloud(centroid_cloud_ptr);

    for (std::unordered_map<Id, Segment>::iterator it = cloud.begin();
        it != cloud.end(); ++it) {

      // If this id is not already in the list to be removed.
      if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                    it->second.segment_id) == duplicate_segments_ids.end()) {

        if (it->second.empty()) continue;

        std::vector<int> nearest_neighbour_indice(n_nearest_segments);
        std::vector<float> nearest_neighbour_squared_distance(n_nearest_segments);

        // Find the nearest neighbours.
        if (kdtree.nearestKSearch(it->second.getLastView().centroid,
                                  n_nearest_segments, nearest_neighbour_indice,
                                  nearest_neighbour_squared_distance) <= 0) {
          LOG(ERROR) << "Nearest neighbour search failed.";
        }

        for (unsigned int i = 1u; i < n_nearest_segments; ++i) {
          // Check if within distance.
          if (nearest_neighbour_squared_distance[i] <= minimum_distance_squared) {
            Segment* other_segment;
            cloud.findValidSegmentPtrById(
                segment_ids[nearest_neighbour_indice[i]], &other_segment);

            // Keep the oldest segment.
            // But keep newest features.
            const size_t min_time_between_segments_for_removing = 20000000000u;
            Id id_to_remove;
            if (it->second.getLastView().timestamp_ns != other_segment->getLastView().timestamp_ns) {
              if (it->second.getLastView().timestamp_ns > other_segment->getLastView().timestamp_ns) {
                if (it->second.track_id != other_segment->track_id &&
                    it->second.getLastView().timestamp_ns < other_segment->getLastView().timestamp_ns + min_time_between_segments_for_removing) {
                  continue;
                }

                id_to_remove = it->second.segment_id;

                // Add id to remove if not already in the list.
                if (std::find(duplicate_segments_ids.begin(), duplicate_segments_ids.end(),
                              id_to_remove) == duplicate_segments_ids.end()) {
                  duplicate_segments_ids.push_back(id_to_remove);

                }
                other_segment->getLastView().features = it->second.getLastView().features;
                break;
              } else {
                if (it->second.track_id != other_segment->track_id &&
                    other_segment->getLastView().timestamp_ns < it->second.getLastView().timestamp_ns + min_time_between_segments_for_removing) {
                  continue;
                }

                id_to_remove = other_segment->segment_id;
                it->second.getLastView().features = other_segment->getLastView().features;
              }
            } else if (it->second.getLastView().point_cloud.size()
                > other_segment->getLastView().point_cloud.size()) {
              id_to_remove = other_segment->segment_id;
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
  cloud.deleteSegmentsById(duplicate_segments_ids, &n_removals);
}

void SegMatch::displayTimings() const {
  Benchmarker::logStatistics(LOG(INFO));
}

void SegMatch::saveTimings() const {
  Benchmarker::saveData();

  Eigen::MatrixXd matrix;
  matrix.resize(loops_timestamps_.size(), 1);
  for (size_t i = 0u; i < loops_timestamps_.size(); ++i) {
    matrix(i,0) = loops_timestamps_[i];
  }
  writeEigenMatrixXdCSV(matrix, "/tmp/timing_loops.csv");
}

} // namespace segmatch
