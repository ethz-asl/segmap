#include "segmatch_ros/segmatch_worker.hpp"

#include <laser_slam/common.hpp>

namespace segmatch_ros {

using namespace laser_slam;
using namespace segmatch;

SegMatchWorker::SegMatchWorker(){ }

SegMatchWorker::~SegMatchWorker() { }

void SegMatchWorker::init(ros::NodeHandle& nh, const SegMatchWorkerParams& params,
                          unsigned int num_tracks) {
  params_ = params;

  // Initialize SegMatch.
  segmatch_.init(params_.segmatch_params, num_tracks);

  // Setup publishers.
  source_representation_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/source_representation", kPublisherQueueSize);
  target_representation_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/target_representation", kPublisherQueueSize);
  matches_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/segmatch/segment_matches", kPublisherQueueSize);
  loop_closures_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/segmatch/loop_closures", kPublisherQueueSize);
  segmentation_positions_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/segmentation_positions", kPublisherQueueSize);
  target_segments_centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/target_segments_centroids", kPublisherQueueSize);
  source_segments_centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/source_segments_centroids", kPublisherQueueSize);

  if (params_.localize) {
    loadTargetCloud();
    publishTargetRepresentation();
    publishTargetSegmentsCentroids();
  }
}

void SegMatchWorker::loadTargetCloud() {
  ROS_INFO("Loading target cloud.");
  PointICloud target_cloud;
  segmatch::loadCloud(params_.target_cloud_filename, &target_cloud);
  segmatch_.processAndSetAsTargetCloud(target_cloud);
  target_cloud_loaded_ = true;
}

bool SegMatchWorker::processSourceCloud(const PointICloud& source_cloud,
                                        const laser_slam::Pose& latest_pose,
                                        unsigned int track_id,
                                        RelativePose* loop_closure) {
  if(params_.close_loops) {
    CHECK_NOTNULL(loop_closure);
  }
  bool loop_closure_found = false;

  if ((params_.localize && target_cloud_loaded_) || params_.close_loops) {
    laser_slam::Clock clock;

    // Check that the robot drove enough since last segmentation.
    bool robot_drove_enough = false;
    if (last_segmented_poses_.empty()) {
      last_segmented_poses_.push_back(PoseTrackIdPair(latest_pose, track_id));
      robot_drove_enough = true;
    } else {
      bool last_segmented_pose_set = false;
      for (auto& pose_track_id_pair: last_segmented_poses_) {
        if (pose_track_id_pair.second == track_id) {
          last_segmented_pose_set = true;
          if (distanceBetweenTwoSE3(pose_track_id_pair.first.T_w, latest_pose.T_w) >
          params_.distance_between_segmentations_m) {
            robot_drove_enough = true;
            pose_track_id_pair.first = latest_pose;
          }
        }
      }
      if (!last_segmented_pose_set) {
        robot_drove_enough = true;
        last_segmented_poses_.push_back(PoseTrackIdPair(latest_pose, track_id));
      }
    }

    if (robot_drove_enough) {
      // Process the source cloud.
      clock.start();
      segmatch_.processAndSetAsSourceCloud(source_cloud, latest_pose, track_id);
      LOG(INFO) << "Processing the source cloud took " << clock.takeRealTime() << " ms.";

      // Find matches.
      clock.start();
      PairwiseMatches predicted_matches = segmatch_.findMatches();
      LOG(INFO) << "Finding matches took " << clock.takeRealTime() << " ms.";
      if (!predicted_matches.empty()) {
        LOG(INFO) << "Number of candidates after full matching: " << predicted_matches.size() <<
                        ".";
      }

      // Filter matches.
      clock.start();
      PairwiseMatches filtered_matches;
      loop_closure_found = segmatch_.filterMatches(predicted_matches, &filtered_matches,
                                                   loop_closure);
      LOG(INFO) << "Filtering matches took " << clock.takeRealTime() << " ms.";

      // TODO move after optimizing and updating target map?
      if (params_.close_loops) {
        // If we did not find a loop-closure, transfer the source to the target map.
        if (filtered_matches.empty()) {
          LOG(INFO) << "Transfering source cloud to target.";
          segmatch_.transferSourceToTarget();
        }
      } else if (params_.localize){
        if (!filtered_matches.empty() && !first_localization_occured) {
          first_localization_occured = true;
          if (params_.align_target_map_on_first_loop_closure) {
            LOG(INFO) << "Aligning target map.";
            segmatch_.alignTargetMap();
            publishTargetRepresentation();
          }
        }
      }

      if (params_.localize) {
        if (!first_localization_occured || !filtered_matches.empty()) {
          publish();
        }
      }

      if (params_.close_loops && filtered_matches.empty()) {
        publish();
      }
    }
  }
  return loop_closure_found;
}

void SegMatchWorker::update(const Trajectory& trajectory) {
  std::vector<Trajectory> trajectories;
  trajectories.push_back(trajectory);
  segmatch_.update(trajectories);
  publish();
}

void SegMatchWorker::update(const std::vector<Trajectory>& trajectories) {
  segmatch_.update(trajectories);
  publish();
}

void SegMatchWorker::publish() const {
  // Publish matches, source representation and segmentation positions.
  publishMatches();
  publishSourceRepresentation();
  publishSourceSegmentsCentroids();
  publishSegmentationPositions();
  // If closing loops, republish the target map.
  if (params_.close_loops) {
    publishTargetRepresentation();
    publishLoopClosures();
    publishTargetSegmentsCentroids();
  }
}

void SegMatchWorker::publishTargetRepresentation() const {
  PointICloud target_representation;
  segmatch_.getTargetRepresentation(&target_representation);
  translateCloud(Translation(0.0, 0.0, -params_.distance_to_lower_target_cloud_for_viz_m),
                 &target_representation);
  applyRandomFilterToCloud(params_.ratio_of_points_to_keep_when_publishing,
                           &target_representation);
  sensor_msgs::PointCloud2 target_representation_as_message;
  convert_to_point_cloud_2_msg(target_representation, params_.world_frame,
                               &target_representation_as_message);
  target_representation_pub_.publish(target_representation_as_message);
}

void SegMatchWorker::publishSourceRepresentation() const {
  PointICloud source_representation;
  segmatch_.getSourceRepresentation(&source_representation);
  applyRandomFilterToCloud(params_.ratio_of_points_to_keep_when_publishing,
                           &source_representation);
  sensor_msgs::PointCloud2 source_representation_as_message;
  convert_to_point_cloud_2_msg(source_representation, params_.world_frame,
                               &source_representation_as_message);
  source_representation_pub_.publish(source_representation_as_message);
}

void SegMatchWorker::publishMatches() const {
  const PairwiseMatches matches = segmatch_.getFilteredMatches();
  PointPairs point_pairs;
  for (size_t i = 0u; i < matches.size(); ++i) {
    PclPoint target_segment_centroid = matches[i].getCentroids().second;
    target_segment_centroid.z -= params_.distance_to_lower_target_cloud_for_viz_m;
    point_pairs.push_back(
        PointPair(matches[i].getCentroids().first, target_segment_centroid));
  }
  publishLineSet(point_pairs, params_.world_frame, kLineScaleSegmentMatches,
                 Color(0.0, 1.0, 0.0), matches_pub_);
}

void SegMatchWorker::publishSegmentationPositions() const {
  PointCloud segmentation_positions_cloud;
  std::vector<Trajectory> segmentation_poses;
  segmatch_.getSegmentationPoses(&segmentation_poses);

  for (const auto& trajectory: segmentation_poses) {
    for (const auto& pose: trajectory) {
      segmentation_positions_cloud.points.push_back(
          se3ToPclPoint(pose.second));
    }
  }

  segmentation_positions_cloud.width = 1;
  segmentation_positions_cloud.height = segmentation_positions_cloud.points.size();
  sensor_msgs::PointCloud2 segmentation_positions_as_message;
  convert_to_point_cloud_2_msg(segmentation_positions_cloud, params_.world_frame,
                               &segmentation_positions_as_message);
  segmentation_positions_pub_.publish(segmentation_positions_as_message);
}

void SegMatchWorker::publishTargetSegmentsCentroids() const {
  PointICloud segments_centroids;
  segmatch_.getTargetSegmentsCentroids(&segments_centroids);
  translateCloud(Translation(0.0, 0.0, -params_.distance_to_lower_target_cloud_for_viz_m),
                 &segments_centroids);
  sensor_msgs::PointCloud2 segments_centroids_as_message;
  convert_to_point_cloud_2_msg(segments_centroids, params_.world_frame,
                               &segments_centroids_as_message);
  target_segments_centroids_pub_.publish(segments_centroids_as_message);
}

void SegMatchWorker::publishSourceSegmentsCentroids() const {
  PointICloud segments_centroids;
  segmatch_.getSourceSegmentsCentroids(&segments_centroids);
  sensor_msgs::PointCloud2 segments_centroids_as_message;
  convert_to_point_cloud_2_msg(segments_centroids, params_.world_frame,
                               &segments_centroids_as_message);
  source_segments_centroids_pub_.publish(segments_centroids_as_message);
}

void SegMatchWorker::publishLoopClosures() const {
  PointPairs point_pairs;

  std::vector<laser_slam::RelativePose> loop_closures;
  segmatch_.getLoopClosures(&loop_closures);
  std::vector<laser_slam::Trajectory> segmentation_poses;
  segmatch_.getSegmentationPoses(&segmentation_poses);

  for (const auto& loop_closure: loop_closures) {
    PclPoint source_point, target_point;
    source_point = se3ToPclPoint(
        segmentation_poses.at(loop_closure.track_id_a).at(loop_closure.time_a_ns));
    target_point = se3ToPclPoint(
        segmentation_poses.at(loop_closure.track_id_b).at(loop_closure.time_b_ns));
    point_pairs.push_back(PointPair(source_point, target_point));
  }

  // Query the segmentation_poses_ at that time.
  publishLineSet(point_pairs, params_.world_frame, kLineScaleLoopClosures,
                 Color(0.0, 0.0, 1.0), loop_closures_pub_);
}

} // namespace segmatch_ros
