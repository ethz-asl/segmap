#include "segmapper/segmapper.hpp"

#include <stdlib.h>

#include <laser_slam/benchmarker.hpp>
#include <laser_slam/common.hpp>
#include <laser_slam_ros/common.hpp>
#include <ros/ros.h>
#include <segmatch/utilities.hpp>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

using namespace laser_slam;
using namespace laser_slam_ros;
using namespace segmatch;
using namespace segmatch_ros;

SegMapper::SegMapper(ros::NodeHandle& n) : nh_(n) {
  // Load ROS parameters from server.
  getParameters();

  // TODO: it would be great to have a cleaner check here, e.g. by having the segmenter interface
  // telling us if normals are needed or not. Unfortunately, at the moment the segmenters are
  // created much later ...
  const std::string& segmenter_type =
      segmatch_worker_params_.segmatch_params.segmenter_params.segmenter_type;
  const bool needs_normal_estimation =
      (segmenter_type == "SimpleSmoothnessConstraints") ||
      (segmenter_type == "IncrementalSmoothnessConstraints");

  // Configure benchmarker
  Benchmarker::setParameters(benchmarker_params_);

  // Create an incremental estimator.
  std::shared_ptr<IncrementalEstimator> incremental_estimator(
      new IncrementalEstimator(params_.online_estimator_params, params_.number_of_robots));

  incremental_estimator_ = incremental_estimator;

  // Create local map publisher
  local_maps_mutexes_ = std::vector<std::mutex>(params_.number_of_robots);
  if (laser_slam_worker_params_.publish_local_map) {
    local_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        laser_slam_worker_params_.local_map_pub_topic,
        kPublisherQueueSize);
  }

  // Setup the laser_slam workers.
  ROS_INFO_STREAM("Number of laser_slam workers: " << params_.number_of_robots);
  for (unsigned int i = 0u; i < params_.number_of_robots; ++i) {
    // Adjust the topics and frames for that laser_slam worker.
    LaserSlamWorkerParams params = laser_slam_worker_params_;

    // Create a local map for each robot.
    std::unique_ptr<NormalEstimator> normal_estimator = nullptr;
    if (needs_normal_estimation) {
      normal_estimator = NormalEstimator::create(
          segmatch_worker_params_.segmatch_params.normal_estimator_type,
          segmatch_worker_params_.segmatch_params.radius_for_normal_estimation_m);
    }
    local_maps_.emplace_back(
        segmatch_worker_params_.segmatch_params.local_map_params, std::move(normal_estimator));

    // TODO rm offset when updating mr_foundry.
    const unsigned int offset = 0;
    if (params_.number_of_robots > 1) {
      // Subscribers.

      params.assembled_cloud_sub_topic = "/" + params_.robot_prefix + std::to_string(i + offset) +
          "/" + laser_slam_worker_params_.assembled_cloud_sub_topic;

      // TF frames.
      params.odom_frame =  params_.robot_prefix + std::to_string(i + offset) +
          "/" + laser_slam_worker_params_.odom_frame;
      params.sensor_frame =  params_.robot_prefix + std::to_string(i + offset) +
          "/" + laser_slam_worker_params_.sensor_frame;

      // Publishers.
      params.trajectory_pub_topic = params_.robot_prefix + std::to_string(i + offset) + "/" +
          laser_slam_worker_params_.trajectory_pub_topic;

      params.local_map_pub_topic = params_.robot_prefix + std::to_string(i + offset) + "/" +
          laser_slam_worker_params_.local_map_pub_topic;
    }

    LOG(INFO) << "Robot " << i << " subscribes to " << params.assembled_cloud_sub_topic << " "
        << params.odom_frame << " and " << params.sensor_frame;

    LOG(INFO) << "Robot " << i << " publishes to " << params.trajectory_pub_topic << " and "
        << params.local_map_pub_topic;

    std::unique_ptr<LaserSlamWorker> laser_slam_worker(new LaserSlamWorker());
    laser_slam_worker->init(nh_, params, incremental_estimator_, i);
    laser_slam_workers_.push_back(std::move(laser_slam_worker));
  }

  // Advertise the save_map service.
  save_map_ = nh_.advertiseService("save_map", &SegMapper::saveMapServiceCall, this);
  save_local_map_ = nh_.advertiseService("save_local_map", &SegMapper::saveLocalMapServiceCall, this);

  // Initialize the SegMatchWorker.
  if (segmatch_worker_params_.localize || segmatch_worker_params_.close_loops) {
    segmatch_worker_.init(n, segmatch_worker_params_, params_.number_of_robots);
  }
  
  for (size_t i = 0u; i < laser_slam_workers_.size(); ++i) {
      skip_counters_.push_back(0u);
      first_points_received_.push_back(false);
  }
}

SegMapper::~SegMapper() {}

void SegMapper::publishMapThread() {
  // Check if map publication is required.
  if (!laser_slam_worker_params_.publish_local_map)
    return;

  ros::Rate thread_rate(laser_slam_worker_params_.map_publication_rate_hz);
  while (ros::ok()) {
    LOG(INFO) << "publishing local maps";
    MapCloud local_maps;
    for (size_t i = 0u; i < local_maps_.size(); ++i) {
      std::unique_lock<std::mutex> map_lock(local_maps_mutexes_[i]);
      local_maps += local_maps_[i].getFilteredPoints();
      map_lock.unlock();
    }
    sensor_msgs::PointCloud2 msg;
    laser_slam_ros::convert_to_point_cloud_2_msg(
        local_maps,
        params_.world_frame, &msg);
    local_map_pub_.publish(msg);
    thread_rate.sleep();
  }
}

void SegMapper::publishTfThread() {
  if (params_.publish_world_to_odom) {
    ros::Rate thread_rate(params_.tf_publication_rate_hz);
    while (ros::ok()) {
      for (size_t i = 0u; i < laser_slam_workers_.size(); ++i) {
        tf::StampedTransform world_to_odom = laser_slam_workers_[i]->getWorldToOdom();
        world_to_odom.stamp_ = ros::Time::now();
        tf_broadcaster_.sendTransform(world_to_odom);
      }
      thread_rate.sleep();
    }
  }
}

void SegMapper::segMatchThread() {
  // Terminate the thread if localization and loop closure are not needed.
  if ((!segmatch_worker_params_.localize &&
      !segmatch_worker_params_.close_loops) ||
      laser_slam_workers_.empty())
    return;

  unsigned int track_id = laser_slam_workers_.size() - 1u;
  // Number of tracks skipped because waiting for new voxels to activate.
  unsigned int skipped_tracks_count = 0u;
  ros::Duration sleep_duration(kSegMatchSleepTime_s);

  unsigned int n_loops = 0u;

  while (ros::ok()) {
    // If all the tracks have been skipped consecutively, sleep for a bit to
    // free some CPU time.
    if (skipped_tracks_count == laser_slam_workers_.size()) {
      skipped_tracks_count = 0u;
      sleep_duration.sleep();
    }

    // Make sure that all the measurements in this loop iteration will get the same timestamp. This
    // makes it easier to plot the data.
    BENCHMARK_START_NEW_STEP();
    // No, we don't include sleeping in the timing, as it is an intended delay.
    BENCHMARK_START("SM");
    // Set the next source cloud to process.
    track_id = (track_id + 1u) % laser_slam_workers_.size();

    // Get the queued points.
    auto new_points = laser_slam_workers_[track_id]->getQueuedPoints();
    if (new_points.empty()) {
      BENCHMARK_STOP_AND_IGNORE("SM");
      ++skipped_tracks_count;
      // Keep asking for publishing to increase the publishing counter.
      segmatch_worker_.publish();
      continue;
    } else {
        if (!first_points_received_[track_id]) {
            first_points_received_[track_id] = true;
            skip_counters_[track_id] = 0u;
        }
    }

    // Update the local map with the new points and the new pose.
    Pose current_pose = incremental_estimator_->getCurrentPose(track_id);
    {
      std::lock_guard<std::mutex> map_lock(local_maps_mutexes_[track_id]);
      local_maps_[track_id].updatePoseAndAddPoints(new_points, current_pose);
    }

    // Process the source cloud.
    if (segmatch_worker_params_.localize) {
      if (segmatch_worker_.processLocalMap(local_maps_[track_id], current_pose, track_id)) {
        if (!pose_at_last_localization_set_) {
          pose_at_last_localization_set_ = true;
          pose_at_last_localization_ = current_pose.T_w;
        } else {
          BENCHMARK_RECORD_VALUE("SM.LocalizationDistances", distanceBetweenTwoSE3(
              pose_at_last_localization_, current_pose.T_w));
          pose_at_last_localization_ = current_pose.T_w;
        }
      }
    } else {
      RelativePose loop_closure;

      // If there is a loop closure.
      if (segmatch_worker_.processLocalMap(local_maps_[track_id], current_pose,
                                           track_id, &loop_closure)) {
        BENCHMARK_BLOCK("SM.ProcessLoopClosure");
        LOG(INFO)<< "Found loop closure! track_id_a: " << loop_closure.track_id_a <<
            " time_a_ns: " << loop_closure.time_a_ns <<
            " track_id_b: " << loop_closure.track_id_b <<
            " time_b_ns: " << loop_closure.time_b_ns;

        // Prevent the workers to process further scans (and add variables to the graph).
        BENCHMARK_START("SM.ProcessLoopClosure.WaitingForLockOnLaserSlamWorkers");
        for (auto& worker: laser_slam_workers_) {
          worker->setLockScanCallback(true);
        }
        BENCHMARK_STOP("SM.ProcessLoopClosure.WaitingForLockOnLaserSlamWorkers");

        // Save last poses for updating the local maps.
        BENCHMARK_START("SM.ProcessLoopClosure.GettingLastPoseOfTrajectories");
        Trajectory trajectory;
        std::vector<SE3> last_poses_before_update;
        std::vector<laser_slam::Time> last_poses_timestamp_before_update_ns;
        if (!params_.clear_local_map_after_loop_closure) {
          for (const auto& worker: laser_slam_workers_) {
            worker->getTrajectory(&trajectory);
            last_poses_before_update.push_back(trajectory.rbegin()->second);
            last_poses_timestamp_before_update_ns.push_back(trajectory.rbegin()->first);
          }
        }
        BENCHMARK_STOP("SM.ProcessLoopClosure.GettingLastPoseOfTrajectories");

        BENCHMARK_START("SM.ProcessLoopClosure.UpdateIncrementalEstimator");
        incremental_estimator_->processLoopClosure(loop_closure);
        BENCHMARK_STOP("SM.ProcessLoopClosure.UpdateIncrementalEstimator");

        BENCHMARK_START("SM.ProcessLoopClosure.ProcessLocalMap");
        for (size_t i = 0u; i < laser_slam_workers_.size(); ++i) {
          if (!params_.clear_local_map_after_loop_closure) {
            laser_slam::SE3 local_map_update_transform =
                laser_slam_workers_[i]->getTransformBetweenPoses(
                    last_poses_before_update[i], last_poses_timestamp_before_update_ns[i]);
            std::unique_lock<std::mutex> map_lock2(local_maps_mutexes_[i]);
            local_maps_[i].transform(local_map_update_transform.cast<float>());
            map_lock2.unlock();
          } else {
            std::unique_lock<std::mutex> map_lock2(local_maps_mutexes_[i]);
            local_maps_[i].clear();
            map_lock2.unlock();
          }
        }
        BENCHMARK_STOP("SM.ProcessLoopClosure.ProcessLocalMap");

        MapCloud local_maps;
        for (size_t i = 0u; i < local_maps_.size(); ++i) {
          std::unique_lock<std::mutex> map_lock(local_maps_mutexes_[i]);
          local_maps += local_maps_[i].getFilteredPoints();
          map_lock.unlock();
        }
        sensor_msgs::PointCloud2 msg;
        laser_slam_ros::convert_to_point_cloud_2_msg(
            local_maps,
            params_.world_frame, &msg);
        local_map_pub_.publish(msg);

        // Update the Segmatch object.
        std::vector<Trajectory> updated_trajectories;
        for (const auto& worker: laser_slam_workers_) {
          worker->getTrajectory(&trajectory);
          updated_trajectories.push_back(trajectory);
        }

        BENCHMARK_START("SM.ProcessLoopClosure.UpdateSegMatch");
        segmatch_worker_.update(updated_trajectories);
        BENCHMARK_STOP("SM.ProcessLoopClosure.UpdateSegMatch");

        //Publish the trajectories.
        for (const auto& worker : laser_slam_workers_) {
          worker->publishTrajectories();
        }

        // Unlock the workers.
        for (auto& worker: laser_slam_workers_) {
          worker->setLockScanCallback(false);
        }

        n_loops++;
        LOG(INFO) << "That was the loop number " << n_loops << ".";
      }

      for (const auto& worker : laser_slam_workers_) {
        worker->publishTrajectories();
      }
    }

    // The track was processed, reset the counter.
    skipped_tracks_count = 0;
    skip_counters_[track_id] = 0u;
    BENCHMARK_STOP("SM");
  }

  Benchmarker::logStatistics(LOG(INFO));
  Benchmarker::saveData();
}

bool SegMapper::saveMapServiceCall(segmapper::SaveMap::Request& request,
                                   segmapper::SaveMap::Response& response) {
  try {
    pcl::io::savePCDFileASCII(request.filename.data,
                              local_maps_.front().getFilteredPoints());
  }
  catch (const std::runtime_error& e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }
  return true;
}

bool SegMapper::saveLocalMapServiceCall(segmapper::SaveMap::Request& request,
                                        segmapper::SaveMap::Response& response) {
  // TODO this is saving only the local map of worker ID 0.
  std::unique_lock<std::mutex> map_lock(local_maps_mutexes_[0]);
  MapCloud local_map;
  local_map += local_maps_[0].getFilteredPoints();
  map_lock.unlock();
  try {
    pcl::io::savePCDFileASCII(request.filename.data, mapPoint2PointCloud(local_map));
  }
  catch (const std::runtime_error& e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }
  return true;
}

void SegMapper::getParameters() {
  // SegMapper parameters.
  const std::string ns = "/SegMapper";
  nh_.getParam(ns + "/number_of_robots",
               params_.number_of_robots);
  nh_.getParam(ns + "/robot_prefix",
               params_.robot_prefix);

  CHECK_GE(params_.number_of_robots, 0u);

  nh_.getParam(ns + "/publish_world_to_odom",
               params_.publish_world_to_odom);
  nh_.getParam(ns + "/world_frame",
               params_.world_frame);
  nh_.getParam(ns + "/tf_publication_rate_hz",
               params_.tf_publication_rate_hz);

  nh_.getParam(ns + "/clear_local_map_after_loop_closure",
               params_.clear_local_map_after_loop_closure);

  // laser_slam worker parameters.
  laser_slam_worker_params_ = laser_slam_ros::getLaserSlamWorkerParams(nh_, ns);
  laser_slam_worker_params_.world_frame = params_.world_frame;

  // Online estimator parameters.
  params_.online_estimator_params = laser_slam_ros::getOnlineEstimatorParams(nh_, ns);

  // Benchmarker parameters.
  benchmarker_params_ = laser_slam_ros::getBenchmarkerParams(nh_, ns);

  // ICP configuration files.
  nh_.getParam("icp_configuration_file",
               params_.online_estimator_params.laser_track_params.icp_configuration_file);
  nh_.getParam("icp_input_filters_file",
               params_.online_estimator_params.laser_track_params.icp_input_filters_file);

  // SegMatchWorker parameters.
  segmatch_worker_params_ = segmatch_ros::getSegMatchWorkerParams(nh_, ns);
  segmatch_worker_params_.world_frame = params_.world_frame;
}
