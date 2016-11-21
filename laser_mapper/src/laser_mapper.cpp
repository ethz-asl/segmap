#include "laser_mapper/laser_mapper.hpp"

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include <Eigen/Eigenvalues>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <laser_slam_ros/common.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pointmatcher_ros/point_cloud.h>
#include <pointmatcher_ros/transform.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace laser_slam;
using namespace segmatch;
using namespace segmatch_ros;

LaserMapper::LaserMapper(ros::NodeHandle& n) : nh_(n) {
  // Load ROS parameters from server.
  getParameters();

  // Setup subscribers.
  ROS_INFO_STREAM("params_.assembled_cloud_sub_topic" << params_.assembled_cloud_sub_topic);
  scan_sub_ = nh_.subscribe(params_.assembled_cloud_sub_topic, kScanSubscriberMessageQueueSize,
                            &LaserMapper::scanCallback, this);

  // Setup publishers.
  trajectory_pub_ = nh_.advertise<nav_msgs::Path>(params_.trajectory_pub_topic,
                                                  kPublisherQueueSize, true);
  odometry_trajectory_pub_ = nh_.advertise<nav_msgs::Path>(params_.odometry_trajectory_pub_topic,
                                                           kPublisherQueueSize, true);
  if (params_.publish_full_map) {
    point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(params_.full_map_pub_topic,
                                                               kPublisherQueueSize);
  }

  if (params_.publish_local_map) {
    local_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(params_.local_map_pub_topic,
                                                             kPublisherQueueSize);
  }

  if (params_.publish_distant_map) {
    distant_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(params_.distant_map_pub_topic,
                                                               kPublisherQueueSize);
  }

  new_fixed_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("new_fixed_cloud",
                                                                 kPublisherQueueSize);

  // Create a sliding-window estimator.
  incremental_estimator_mutex_.lock();
  incremental_estimator_ = IncrementalEstimator(params_.online_estimator_params);
  incremental_estimator_mutex_.unlock();

  // Advertise the save_map and save_distant_map services.
  save_map_ = nh_.advertiseService("save_map", &LaserMapper::saveMapServiceCall, this);
  save_distant_map_ = nh_.advertiseService("save_distant_map",
                                           &LaserMapper::saveDistantMapServiceCall, this);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(4, 4);
  matrix = Eigen::Matrix<float, 4,4>::Identity();

  world_to_odom_mutex_.lock();
  world_to_odom_ = PointMatcher_ros::eigenMatrixToStampedTransform<float>(
      matrix, params_.world_frame, params_.odom_frame, ros::Time::now());
  world_to_odom_mutex_.unlock();

  voxel_filter_.setLeafSize(params_.voxel_size_m, params_.voxel_size_m,
                            params_.voxel_size_m);
  voxel_filter_.setMinimumPointsNumberPerVoxel(params_.minimum_point_number_per_voxel);

  // Initialize the SegMatchWorker.
  if (segmatch_worker_params_.localize || segmatch_worker_params_.close_loops) {
    segmatch_worker_.init(n, segmatch_worker_params_);
  }
}

LaserMapper::~LaserMapper() {}

void LaserMapper::publishMapThread() {
  if (params_.create_filtered_map) {
    ros::Rate thread_rate(params_.map_publication_rate_hz);
    while (ros::ok()) {
      if (local_map_.size() > 0u) {
        publishMap();
      }
      thread_rate.sleep();
    }
  }
}

void LaserMapper::publishTfThread() {
  if (params_.publish_world_to_odom) {
    ros::Rate thread_rate(params_.tf_publication_rate_hz);
    while (ros::ok()) {
      world_to_odom_mutex_.lock();
      world_to_odom_.stamp_ = ros::Time::now();
      tf_broadcaster_.sendTransform(world_to_odom_);
      world_to_odom_mutex_.unlock();
      thread_rate.sleep();
    }
  }
}

void LaserMapper::segMatchThread() {
  if (segmatch_worker_params_.localize || segmatch_worker_params_.close_loops) {
    ros::Rate thread_rate(kSegMatchThreadRate);
    while (ros::ok()) {
      if (source_cloud_ready_) {
        source_cloud_ready_mutex_.lock();
        source_cloud_ready_ = false;
        source_cloud_ready_mutex_.unlock();

        // Get current pose.
        incremental_estimator_mutex_.lock();
        Pose current_pose = incremental_estimator_.getCurrentPose();
        incremental_estimator_mutex_.unlock();

        // Get source cloud.
        local_map_filtered_mutex_.lock();
        segmatch::PointICloud source_cloud;
        pcl::copyPointCloud(local_map_filtered_, source_cloud);
        local_map_filtered_mutex_.unlock();

        // Process the source cloud.
        if (segmatch_worker_params_.localize) {
          segmatch_worker_.processSourceCloud(source_cloud, current_pose);
        } else {
          RelativePose loop_closure;
          // If there is a loop closure.
          if (segmatch_worker_.processSourceCloud(source_cloud, current_pose, &loop_closure)) {
            Trajectory new_traj;
            incremental_estimator_mutex_.lock();
            incremental_estimator_.processLoopClosure(loop_closure);
            incremental_estimator_.getTrajectory(&new_traj);
            incremental_estimator_mutex_.unlock();

            // Clear the local map if desired.
            if (params_.clear_local_map_after_loop_closure) {
              local_map_mutex_.lock();
              local_map_.clear();
              local_map_mutex_.unlock();
            }

            // Update the Segmatch object.
            segmatch_worker_.update(new_traj);
          }
        }
      }
      thread_rate.sleep();
    }
  }
}

void LaserMapper::scanCallback(const sensor_msgs::PointCloud2& cloud_msg_in) {
  if (tf_listener_.waitForTransform(params_.odom_frame, params_.sensor_frame,
                                    cloud_msg_in.header.stamp, ros::Duration(kTimeout_s))) {
    // Get the tf transform.
    tf::StampedTransform tf_transform;
    tf_listener_.lookupTransform(params_.odom_frame, params_.sensor_frame,
                                 cloud_msg_in.header.stamp, tf_transform);

    bool process_scan = false;
    SE3 current_pose;

    // Convert input cloud to laser scan.
    LaserScan new_scan;
    new_scan.scan = PointMatcher_ros::rosMsgToPointMatcherCloud<float>(cloud_msg_in);
    new_scan.time_ns = rosTimeToCurveTime(cloud_msg_in.header.stamp.toNSec());

    if (!last_pose_set_) {
      process_scan = true;
      last_pose_set_ = true;
      last_pose_ = tfTransformToPose(tf_transform).T_w;
    } else {
      current_pose = tfTransformToPose(tf_transform).T_w;
      float dist_m = distanceBetweenTwoSE3(current_pose, last_pose_);
      if (dist_m > params_.minimum_distance_to_add_pose) {
        process_scan = true;
        last_pose_ = current_pose;
      }
    }

    if (process_scan) {
      // Convert input cloud to laser scan.
      LaserScan new_scan;
      new_scan.scan = PointMatcher_ros::rosMsgToPointMatcherCloud<float>(cloud_msg_in);
      new_scan.time_ns = rosTimeToCurveTime(cloud_msg_in.header.stamp.toNSec());

      // Process the pose and the laser scan.
      incremental_estimator_mutex_.lock();
      incremental_estimator_.processPoseAndLaserScan(tfTransformToPose(tf_transform), new_scan);
      incremental_estimator_mutex_.unlock();

      // Adjust the correction between the world and odom frames.
      incremental_estimator_mutex_.lock();
      Pose current_pose = incremental_estimator_.getCurrentPose();
      incremental_estimator_mutex_.unlock();

      SE3 T_odom_sensor = tfTransformToPose(tf_transform).T_w;
      SE3 T_w_sensor = current_pose.T_w;
      SE3 T_w_odom = T_w_sensor * T_odom_sensor.inverse();

      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix;
      // TODO resize needed?
      matrix.resize(4, 4);
      matrix = T_w_odom.getTransformationMatrix().cast<float>();

      world_to_odom_mutex_.lock();
      world_to_odom_ = PointMatcher_ros::eigenMatrixToStampedTransform<float>(
          matrix, params_.world_frame, params_.odom_frame, cloud_msg_in.header.stamp);
      tf_broadcaster_.sendTransform(world_to_odom_);
      world_to_odom_mutex_.unlock();

      publishTrajectories();

      // Publish the local scans which are completely estimated and static.
      laser_slam::DataPoints new_fixed_cloud;
      incremental_estimator_mutex_.lock();
      incremental_estimator_.appendFixedScans(&new_fixed_cloud);
      incremental_estimator_mutex_.unlock();

      //TODO(Renaud) move to a transformPointCloud() fct.
      laser_slam::PointMatcher::TransformationParameters transformation_matrix =
          T_w_sensor.inverse().getTransformationMatrix().cast<float>();

      laser_slam::correctTransformationMatrix(&transformation_matrix);

      laser_slam::PointMatcher::Transformation* rigid_transformation =
          laser_slam::PointMatcher::get().REG(Transformation).create("RigidTransformation");
      CHECK_NOTNULL(rigid_transformation);

      laser_slam::PointMatcher::DataPoints fixed_cloud_in_sensor_frame =
          rigid_transformation->compute(new_fixed_cloud,transformation_matrix);


      new_fixed_cloud_pub_.publish(
          PointMatcher_ros::pointMatcherCloudToRosMsg<float>(fixed_cloud_in_sensor_frame,
                                                             params_.sensor_frame,
                                                             cloud_msg_in.header.stamp));

      PointCloud new_fixed_cloud_pcl = lpmToPcl(new_fixed_cloud);

      // Add the local scans to the full point cloud.
      if (params_.create_filtered_map) {
        if (new_fixed_cloud_pcl.size() > 0u) {
          local_map_mutex_.lock();
          if (local_map_.size() > 0u) {
            local_map_ += new_fixed_cloud_pcl;
            ROS_INFO_STREAM("Adding new fixed cloud to the local map now with size " <<
                            local_map_.size() << ".");
          } else {
            ROS_INFO("Creating a new local map from the fixed cloud.");
            local_map_ = new_fixed_cloud_pcl;
          }
          local_map_mutex_.unlock();
        }
      }

    } else {
      ROS_WARN("[LaserMapper] Scan not processed (Not moved enough since last pose).");
    }
  } else {
    ROS_WARN_STREAM("[LaserMapper] Timeout while waiting between " + params_.odom_frame  +
                    " and " + params_.sensor_frame  + ".");
  }
}


bool LaserMapper::saveMapServiceCall(laser_mapper::SaveMap::Request& request,
                                     laser_mapper::SaveMap::Response& response) {
  PointCloud filtered_map;
  getFilteredMap(&filtered_map);

  try {
    pcl::io::savePCDFileASCII(request.filename.data, filtered_map);
  }
  catch (const std::runtime_error& e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }
  return true;
}

bool LaserMapper::saveDistantMapServiceCall(laser_mapper::SaveMap::Request& request,
                                            laser_mapper::SaveMap::Response& response) {
  PointCloud filtered_map;
  getFilteredMap(&filtered_map);

  try {
    distant_map_mutex_.lock();
    pcl::io::savePCDFileASCII(request.filename.data, distant_map_);
    distant_map_mutex_.unlock();
  }
  catch (const std::runtime_error& e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }
  return true;
}

void LaserMapper::publishTrajectory(const Trajectory& trajectory,
                                    const ros::Publisher& publisher) const {
  nav_msgs::Path traj_msg;
  traj_msg.header.frame_id = params_.world_frame;
  Time traj_time = curveTimeToRosTime(trajectory.rbegin()->first);
  traj_msg.header.stamp.fromNSec(traj_time);

  for (const auto& timePose : trajectory) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = traj_msg.header;
    pose_msg.header.stamp.fromNSec(curveTimeToRosTime(timePose.first));

    //TODO functionize
    pose_msg.pose.position.x = timePose.second.getPosition().x();
    pose_msg.pose.position.y = timePose.second.getPosition().y();
    pose_msg.pose.position.z = timePose.second.getPosition().z();
    pose_msg.pose.orientation.w = timePose.second.getRotation().w();
    pose_msg.pose.orientation.x = timePose.second.getRotation().x();
    pose_msg.pose.orientation.y = timePose.second.getRotation().y();
    pose_msg.pose.orientation.z = timePose.second.getRotation().z();
    traj_msg.poses.push_back(pose_msg);
  }
  publisher.publish(traj_msg);
}

void LaserMapper::publishMap() {
  if (local_map_.size() > 0) {
    PointCloud filtered_map;
    getFilteredMap(&filtered_map);

    // Indicate that a new source cloud is ready to be used for localization and loop-closure.
    source_cloud_ready_mutex_.lock();
    source_cloud_ready_ = true;
    source_cloud_ready_mutex_.unlock();

    //maximumNumberPointsFilter(&filtered_map);
    if (params_.publish_full_map) {
      sensor_msgs::PointCloud2 msg;
      convert_to_point_cloud_2_msg(filtered_map, params_.world_frame, &msg);
      point_cloud_pub_.publish(msg);
    }
    if (params_.publish_local_map) {
      sensor_msgs::PointCloud2 msg;
      local_map_filtered_mutex_.lock();
      convert_to_point_cloud_2_msg(local_map_filtered_, params_.world_frame, &msg);
      local_map_pub_.publish(msg);
      local_map_filtered_mutex_.unlock();
    }
    if (params_.publish_distant_map) {
      distant_map_mutex_.lock();
      sensor_msgs::PointCloud2 msg;
      convert_to_point_cloud_2_msg(distant_map_, params_.world_frame, &msg);
      distant_map_pub_.publish(msg);
      distant_map_mutex_.unlock();
    }
  }
}

void LaserMapper::publishTrajectories() {
  Trajectory trajectory;
  incremental_estimator_mutex_.lock();
  incremental_estimator_.getTrajectory(&trajectory);
  incremental_estimator_mutex_.unlock();
  publishTrajectory(trajectory, trajectory_pub_);
  incremental_estimator_mutex_.lock();
  incremental_estimator_.getOdometryTrajectory(&trajectory);
  incremental_estimator_mutex_.unlock();
  publishTrajectory(trajectory, odometry_trajectory_pub_);
}

// TODO can we move?
Pose LaserMapper::tfTransformToPose(const tf::StampedTransform& tf_transform) {
  // Register new pose.
  Pose pose;
  SE3::Position pos(tf_transform.getOrigin().getX(), tf_transform.getOrigin().getY(),
                    tf_transform.getOrigin().getZ());
  SE3::Rotation::Implementation rot(tf_transform.getRotation().getW(),
                                    tf_transform.getRotation().getX(),
                                    tf_transform.getRotation().getY(),
                                    tf_transform.getRotation().getZ());
  pose.T_w = SE3(pos, rot);
  pose.time_ns = rosTimeToCurveTime(tf_transform.stamp_.toNSec());

  return pose;
}

Time LaserMapper::rosTimeToCurveTime(const Time& timestamp_ns) {
  if (!base_time_set_) {
    base_time_ns_ = timestamp_ns;
    base_time_set_ = true;
  }
  return timestamp_ns - base_time_ns_;
}

Time LaserMapper::curveTimeToRosTime(const Time& timestamp_ns) const {
  CHECK(base_time_set_);
  return timestamp_ns + base_time_ns_;
}

// TODO one shot of cleaning.
void LaserMapper::getFilteredMap(PointCloud* filtered_map) {
  incremental_estimator_mutex_.lock();
  laser_slam::Pose current_pose = incremental_estimator_.getCurrentPose();
  incremental_estimator_mutex_.unlock();

  PclPoint current_position;
  current_position.x = current_pose.T_w.getPosition()[0];
  current_position.y = current_pose.T_w.getPosition()[1];
  current_position.z = current_pose.T_w.getPosition()[2];

  // Apply the cylindrical filter on the local map and get a copy.
  local_map_mutex_.lock();
  PointCloud local_map = local_map_;
  applyCylindricalFilter(current_position, params_.distance_to_consider_fixed,
                         40, false, &local_map_);
  local_map_mutex_.unlock();

  // Apply a voxel filter.
  laser_slam::Clock clock;

  PointCloudPtr local_map_ptr(new PointCloud());
  pcl::copyPointCloud<PclPoint, PclPoint>(local_map, *local_map_ptr);

  PointCloud local_map_filtered;

  voxel_filter_.setInputCloud(local_map_ptr);
  voxel_filter_.filter(local_map_filtered);

  clock.takeTime();

  if (params_.separate_distant_map) {
    // If separating the map is enabled, the distance between each point in the local_map_ will
    // be compared to the current robot position. Points which are far from the robot will
    // be transfered to the distant_map_. This is helpful for publishing (points in distant_map_
    // need to be filtered only once) and for any other processing which needs to be done only
    // when a map is distant from robot and can be assumed as static (until loop closure).

    // TODO(renaud) Is there a way to separate the cloud without having to transform in sensor
    // frame by setting the position to compute distance from?
    // Transform local_map_ in sensor frame.
    clock.start();

    // Save before removing points.
    PointCloud new_distant_map = local_map_filtered;

    applyCylindricalFilter(current_position, params_.distance_to_consider_fixed,
                           40, false, &local_map_filtered);

    applyCylindricalFilter(current_position, params_.distance_to_consider_fixed,
                           40, true, &new_distant_map);

    local_map_filtered_mutex_.lock();
    local_map_filtered_ = local_map_filtered;
    local_map_filtered_mutex_.unlock();

    // Add the new_distant_map to the distant_map_.
    distant_map_mutex_.lock();
    if (distant_map_.size() > 0u) {
      distant_map_ += new_distant_map;
    } else {
      distant_map_ = new_distant_map;
    }

    *filtered_map = local_map_filtered;
    *filtered_map += distant_map_;
    distant_map_mutex_.unlock();

    clock.takeTime();
    // LOG(INFO) << "new_local_map.size() " << local_map.size();
    // LOG(INFO) << "new_distant_map.size() " << new_distant_map.size();
    // LOG(INFO) << "distant_map_.size() " << distant_map_.size();
    // LOG(INFO) << "Separating done! Took " << clock.getRealTime() << " ms.";
  } else {
    *filtered_map = local_map;
  }
}

void LaserMapper::getParameters() {
  // LaserMapper parameters.
  nh_.getParam("/LaserMapper/distance_to_consider_fixed",
               params_.distance_to_consider_fixed);
  nh_.getParam("/LaserMapper/separate_distant_map",
               params_.separate_distant_map);

  nh_.getParam("/LaserMapper/publish_local_map",
               params_.publish_local_map);
  nh_.getParam("/LaserMapper/publish_full_map",
               params_.publish_full_map);
  nh_.getParam("/LaserMapper/publish_distant_map",
               params_.publish_distant_map);
  nh_.getParam("/LaserMapper/map_publication_rate_hz",
               params_.map_publication_rate_hz);

  nh_.getParam("/LaserMapper/publish_world_to_odom",
               params_.publish_world_to_odom);
  nh_.getParam("/LaserMapper/tf_publication_rate_hz",
               params_.tf_publication_rate_hz);

  nh_.getParam("/LaserMapper/assembled_cloud_sub_topic",
               params_.assembled_cloud_sub_topic);
  nh_.getParam("/LaserMapper/trajectory_pub_topic",
               params_.trajectory_pub_topic);
  nh_.getParam("/LaserMapper/odometry_trajectory_pub_topic",
               params_.odometry_trajectory_pub_topic);
  nh_.getParam("/LaserMapper/full_map_pub_topic",
               params_.full_map_pub_topic);
  nh_.getParam("/LaserMapper/local_map_pub_topic",
               params_.local_map_pub_topic);
  nh_.getParam("/LaserMapper/distant_map_pub_topic",
               params_.distant_map_pub_topic);

  nh_.getParam("/LaserMapper/world_frame",
               params_.world_frame);
  nh_.getParam("/LaserMapper/odom_frame",
               params_.odom_frame);
  nh_.getParam("/LaserMapper/sensor_frame",
               params_.sensor_frame);

  nh_.getParam("/LaserMapper/create_filtered_map",
               params_.create_filtered_map);

  nh_.getParam("/LaserMapper/minimum_distance_to_add_pose",
               params_.minimum_distance_to_add_pose);
  nh_.getParam("/LaserMapper/voxel_size_m",
               params_.voxel_size_m);
  nh_.getParam("/LaserMapper/minimum_point_number_per_voxel",
               params_.minimum_point_number_per_voxel);

  nh_.getParam("/LaserMapper/clear_local_map_after_loop_closure",
               params_.clear_local_map_after_loop_closure);

  // Online estimator parameters.
  params_.online_estimator_params = laser_slam_ros::getOnlineEstimatorParams(nh_, "/LaserMapper");

  // ICP configuration files.
  nh_.getParam("icp_configuration_file",
               params_.online_estimator_params.laser_track_params.icp_configuration_file);
  nh_.getParam("icp_input_filters_file",
               params_.online_estimator_params.laser_track_params.icp_input_filters_file);

  // SegMatchWorker parameters.
  segmatch_worker_params_ = segmatch_ros::getSegMatchWorkerParams(nh_, "/LaserMapper");
  segmatch_worker_params_.world_frame = params_.world_frame;
}
