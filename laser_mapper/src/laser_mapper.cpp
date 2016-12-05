#include "laser_mapper/laser_mapper.hpp"

#include <stdlib.h>

#include <laser_slam_ros/common.hpp>
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

using namespace laser_slam;
using namespace laser_slam_ros;
using namespace segmatch;
using namespace segmatch_ros;

LaserMapper::LaserMapper(ros::NodeHandle& n) : nh_(n) {
  // Load ROS parameters from server.
  getParameters();

  // Create an incremental estimator.
  std::shared_ptr<IncrementalEstimator> incremental_estimator(
      new IncrementalEstimator(params_.online_estimator_params));
  incremental_estimator_ = std::move(incremental_estimator);

  // Setup the laser_slam worker.
  std::unique_ptr<LaserSlamWorker> laser_slam_worker(new LaserSlamWorker());
  laser_slam_worker->init(nh_, laser_slam_worker_params_, incremental_estimator_);
  laser_slam_worker_ = std::move(laser_slam_worker);

  // Initialize the SegMatchWorker.
  if (segmatch_worker_params_.localize || segmatch_worker_params_.close_loops) {
    segmatch_worker_.init(n, segmatch_worker_params_);
  }

  // Advertise the save_map service.
  save_map_ = nh_.advertiseService("save_map", &LaserMapper::saveMapServiceCall, this);
}

LaserMapper::~LaserMapper() {}

void LaserMapper::publishMapThread() {
  if (laser_slam_worker_params_.create_filtered_map) {
    ros::Rate thread_rate(laser_slam_worker_params_.map_publication_rate_hz);
    while (ros::ok()) {
      laser_slam_worker_->publishMap();
      thread_rate.sleep();
    }
  }
}

void LaserMapper::publishTfThread() {
  if (params_.publish_world_to_odom) {
    ros::Rate thread_rate(params_.tf_publication_rate_hz);
    while (ros::ok()) {
      tf::StampedTransform world_to_odom = laser_slam_worker_->getWorldToOdom();
      world_to_odom.stamp_ = ros::Time::now();
      tf_broadcaster_.sendTransform(world_to_odom);
      thread_rate.sleep();
    }
  }
}

void LaserMapper::segMatchThread() {
  if (segmatch_worker_params_.localize || segmatch_worker_params_.close_loops) {
    ros::Rate thread_rate(kSegMatchThreadRate_hz);
    while (ros::ok()) {
      // TODO add function to check if the map was updated before getting?
      segmatch::PointCloud local_map_filtered;
      laser_slam_worker_->getLocalMapFiltered(&local_map_filtered);

      if (local_map_filtered.points.size() > 0) {
        // Get source cloud.
        segmatch::PointICloud source_cloud;
        pcl::copyPointCloud(local_map_filtered, source_cloud);

        // Get current pose.
        // TODO get from LaserSlamWorker.
        Pose current_pose = incremental_estimator_->getCurrentPose();

        // Process the source cloud.
        if (segmatch_worker_params_.localize) {
          segmatch_worker_.processSourceCloud(source_cloud, current_pose);
        } else {
          RelativePose loop_closure;
          // If there is a loop closure.
          if (segmatch_worker_.processSourceCloud(source_cloud, current_pose, 0u,
                                                  &loop_closure)) {
            LOG(INFO) << "Found loop closure! time_a_ns: " << loop_closure.time_a_ns <<
                " time_b_ns: " << loop_closure.time_b_ns;

            incremental_estimator_->processLoopClosure(loop_closure);

            // Clear the local map if desired.
            if (params_.clear_local_map_after_loop_closure) {
              laser_slam_worker_->clearLocalMap();
            }

            // Update the Segmatch object.
            Trajectory trajectory;
            laser_slam_worker_->getTrajectory(&trajectory);
            segmatch_worker_.update(trajectory);
            LOG(INFO) << "SegMatchThread updating segmap done";
          }
        }
      }

      thread_rate.sleep();
    }
  }
}

bool LaserMapper::saveMapServiceCall(laser_mapper::SaveMap::Request& request,
                                     laser_mapper::SaveMap::Response& response) {
  PointCloud filtered_map;
  laser_slam_worker_->getFilteredMap(&filtered_map);
  try {
    pcl::io::savePCDFileASCII(request.filename.data, filtered_map);
  }
  catch (const std::runtime_error& e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }
  return true;
}

void LaserMapper::getParameters() {
  // LaserMapper parameters.
  const std::string ns = "/LaserMapper";

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

  // ICP configuration files.
  nh_.getParam("icp_configuration_file",
               params_.online_estimator_params.laser_track_params.icp_configuration_file);
  nh_.getParam("icp_input_filters_file",
               params_.online_estimator_params.laser_track_params.icp_input_filters_file);

  // SegMatchWorker parameters.
  segmatch_worker_params_ = segmatch_ros::getSegMatchWorkerParams(nh_, ns);
  segmatch_worker_params_.world_frame = params_.world_frame;
}
