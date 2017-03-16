#include "segmatch_ros/segmatch_worker.hpp"

#include <unistd.h>

#include <eigen_conversions/eigen_msg.h>
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
  predicted_matches_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/segmatch/predicted_segment_matches", kPublisherQueueSize);
  loop_closures_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/segmatch/loop_closures", kPublisherQueueSize);
  segmentation_positions_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/segmentation_positions", kPublisherQueueSize);
  target_segments_centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/target_segments_centroids", kPublisherQueueSize);
  source_segments_centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/segmatch/source_segments_centroids", kPublisherQueueSize);
  last_transformation_pub_ = nh.advertise<geometry_msgs::Transform>(
      "/segmatch/last_transformation", 5, true);

  if (params_.export_segments_and_matches) {
    export_run_service_ = nh.advertiseService("export_run",
                                              &SegMatchWorker::exportRunServiceCall, this);
  }
  if (std::find(params_.segmatch_params.descriptors_params.descriptor_types.begin(),
                params_.segmatch_params.descriptors_params.descriptor_types.end(), 
                "Autoencoder") != 
      params_.segmatch_params.descriptors_params.descriptor_types.end() && 
      params_.autoencoder_reconstructor_script_path != "") {
    reconstruct_segments_service_ = nh.advertiseService(
        "reconstruct_segments", &SegMatchWorker::reconstructSegmentsServiceCall, this);
    reconstruction_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
        "/segmatch/reconstruction", kPublisherQueueSize);
  }

  if (params_.localize) {
    loadTargetCloud();
    usleep(1000000);
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
      LOG(INFO) << "Number of matches after filtering: " << filtered_matches.size() << ".";

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

      // Store segments and matches in database if desired, for later export.
      if (params_.export_segments_and_matches) {
        segments_database_ += segmatch_.getSourceAsSegmentedCloud();
        if (loop_closure_found) {
          for (size_t i = 0u; i < filtered_matches.size(); ++i) {
            matches_database_.addMatch(filtered_matches.at(i).ids_.first,
                                       filtered_matches.at(i).ids_.second);
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
  publishLastTransformation();
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
  const PairwiseMatches predicted_matches = segmatch_.getPredictedMatches();
  point_pairs.clear();
  for (size_t i = 0u; i < predicted_matches.size(); ++i) {
    PclPoint target_segment_centroid = predicted_matches[i].getCentroids().second;
    target_segment_centroid.z -= params_.distance_to_lower_target_cloud_for_viz_m;
    point_pairs.push_back(
        PointPair(predicted_matches[i].getCentroids().first, target_segment_centroid));
  }
  publishLineSet(point_pairs, params_.world_frame, kLineScaleSegmentMatches,
                 Color(0.7, 0.7, 0.7), predicted_matches_pub_);
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

void SegMatchWorker::publishLastTransformation() const {
  if (first_localization_occured) {
    Eigen::Affine3d transformation;
    segmatch_.getLastTransform(&(transformation.matrix()));
    geometry_msgs::Transform transform_msg;
    tf::transformEigenToMsg(transformation, transform_msg);
    last_transformation_pub_.publish(transform_msg);
  }
}

bool SegMatchWorker::exportRunServiceCall(std_srvs::Empty::Request& req,
                                         std_srvs::Empty::Response& res) {
  // Get current date.
  const boost::posix_time::ptime time_as_ptime = ros::WallTime::now().toBoost();
  std::string acquisition_time = to_iso_extended_string(time_as_ptime);
  database::exportFeatures("/tmp/online_matcher/run_" + acquisition_time + "_features.csv",
                            segments_database_);
  database::exportSegments("/tmp/online_matcher/run_" + acquisition_time + "_segments.csv",
                            segments_database_);
  database::exportMatches("/tmp/online_matcher/run_" + acquisition_time + "_matches.csv",
                           matches_database_);
  return true;
}

bool SegMatchWorker::reconstructSegmentsServiceCall(std_srvs::Empty::Request& req,
                                                    std_srvs::Empty::Response& res) {
  const std::string kSegmentsFilename = "autoencoder_reconstructor_segments.txt";
  const std::string kFeaturesFilename = "autoencoder_reconstructor_features.txt";
  FILE* script_process_pipe;
  DescriptorsParameters descriptors_params = params_.segmatch_params.descriptors_params;
  const std::string command = descriptors_params.autoencoder_python_env + " -u " +
      params_.autoencoder_reconstructor_script_path + " " +
      descriptors_params.autoencoder_model_path + " " +
      descriptors_params.autoencoder_temp_folder_path + kSegmentsFilename + " " +
      descriptors_params.autoencoder_temp_folder_path + kFeaturesFilename + " " +
      std::to_string(descriptors_params.autoencoder_latent_space_dimension) + " 2>&1";

  LOG(INFO) << "Executing command: $" << command;
  if (!(script_process_pipe = popen(command.c_str(), "r"))) {
    LOG(FATAL) << "Could not execute autoencoder reconstruction command";
  }

  char buff[512];
  while (fgets(buff, sizeof(buff), script_process_pipe) != NULL) {
    LOG(INFO) << buff;
    if (std::string(buff) == "__INIT_COMPLETE__\n") {
      break;
    }
  }

  // Export segmented cloud.
  LOG(INFO) << "Exporting autoencoder features.";
  SegmentedCloud original_target_segments = segmatch_.getTargetAsSegmentedCloud();
  database::exportFeatures(descriptors_params.autoencoder_temp_folder_path + kFeaturesFilename,
                           original_target_segments);
  LOG(INFO) << "Done.";

  // Wait for script to describe segments.
  while (fgets(buff, sizeof(buff), script_process_pipe) != NULL) {
    LOG(INFO) << buff;
    if (std::string(buff) == "__RCST_COMPLETE__\n") {
      break;
    }
  }

  SegmentedCloud reconstructed_target_segments;
  // Import the autoencoder features from file.
  LOG(INFO) << "Importing autoencoder segments.";
  CHECK(database::importSegments(descriptors_params.autoencoder_temp_folder_path + kSegmentsFilename,
                                 &reconstructed_target_segments));
  LOG(INFO) << "Done.";

  pclose(script_process_pipe);

  // Move reconstructions to their centroid locations.
  for (std::unordered_map<Id, Segment>::const_iterator it = original_target_segments.begin();
      it != original_target_segments.end(); ++it) {
    PclPoint centroid = it->second.centroid;
    Segment* segment_ptr;
    CHECK(reconstructed_target_segments.findValidSegmentPtrById(it->first, &segment_ptr));
    translateCloud(Translation(centroid.x, centroid.y, centroid.z), &(segment_ptr->point_cloud));
  }

  // Publish reconstruction
  PointICloud reconstructed_target_representation;
  segmentedCloudToCloud(reconstructed_target_segments, &reconstructed_target_representation);
  translateCloud(Translation(0.0, 0.0, -2*params_.distance_to_lower_target_cloud_for_viz_m),
                 &reconstructed_target_representation);
  sensor_msgs::PointCloud2 reconstructed_target_representation_as_message;
  convert_to_point_cloud_2_msg(reconstructed_target_representation, params_.world_frame,
                               &reconstructed_target_representation_as_message);
  reconstruction_pub_.publish(reconstructed_target_representation_as_message);

  return true;
}

} // namespace segmatch_ros
