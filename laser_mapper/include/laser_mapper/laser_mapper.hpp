#ifndef LASER_MAPPER_LASER_MAPPER_HPP_
#define LASER_MAPPER_LASER_MAPPER_HPP_

#include <mutex>
#include <string>
#include <vector>

#include <laser_slam/parameters.hpp>
#include <laser_slam/sliding_window_estimator.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <segmatch/common.hpp>
#include <segmatch_ros/common.hpp>
#include <segmatch_ros/segmatch_worker.hpp>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>

#include "laser_mapper/SaveMap.h"

struct LaserMapperParams {
  // Map creation & filtering parameters.
  double distance_to_consider_fixed;
  bool separate_distant_map;
  bool create_filtered_map;
  double minimum_distance_to_add_pose;
  double voxel_size_m;
  int minimum_point_number_per_voxel;
  bool clear_local_map_after_loop_closure = true;

  // Trajectory estimator parameters.
  laser_slam::OnlineEstimatorParams online_estimator_params;

  // Frames.
  std::string world_frame;
  std::string odom_frame;
  std::string sensor_frame;

  // Topics.
  std::string assembled_cloud_sub_topic;
  std::string trajectory_pub_topic;
  std::string odometry_trajectory_pub_topic;
  std::string full_map_pub_topic;
  std::string local_map_pub_topic;
  std::string distant_map_pub_topic;

  // Map publication.
  bool publish_local_map;
  bool publish_full_map;
  bool publish_distant_map;
  double map_publication_rate_hz;

  // Enable world to odom.
  bool publish_world_to_odom;
  double tf_publication_rate_hz;

}; // struct LaserMapperParams

class LaserMapper {

 public:
  explicit LaserMapper(ros::NodeHandle& n);
  ~LaserMapper();

  /// \brief A thread function for handling map publishing.
  void publishMapThread();

  /// \brief A thread function for updating the transform between world and odom.
  void publishTfThread();

  /// \brief A thread function for segmenting the map.
  void segMatchThread();

 protected:
  /// \brief Register the local scans to the sliding window estimator.
  void scanCallback(const sensor_msgs::PointCloud2& cloud_msg_in);

  /// \brief Call back of the save_map service.
  bool saveMapServiceCall(laser_mapper::SaveMap::Request& request,
                          laser_mapper::SaveMap::Response& response);

  /// \brief Call back of the save_distant_map service.
  bool saveDistantMapServiceCall(laser_mapper::SaveMap::Request& request,
                                 laser_mapper::SaveMap::Response& response);

  /// \brief Publish the robot trajectory (as path) in ROS.
  void publishTrajectory(const laser_slam::Trajectory& trajectory,
                         const ros::Publisher& publisher) const;

  /// \brief Publish the map.
  void publishMap();

  /// \brief Publish the estimated trajectory and the odometry only based trajectory.
  void publishTrajectories();

 private:
  // Convert a tf::StampedTransform to a laser_slam::Pose.
  laser_slam::Pose tfTransformToPose(const tf::StampedTransform& tf_transform);
  // TODO: common.hpp?
  laser_slam::SE3 geometryMsgTransformToSE3(const geometry_msgs::Transform& transform);
  geometry_msgs::Transform SE3ToGeometryMsgTransform(const laser_slam::SE3& transform);

  // Standardize the time so that the trajectory starts at time 0.
  laser_slam::Time rosTimeToCurveTime(const laser_slam::Time& timestamp_ns);

  // Convert time from trajectory base back to ROS base.
  laser_slam::Time curveTimeToRosTime(const laser_slam::Time& timestamp_ns) const;

  // Get a filtered map and apply map separation if desired.
  void getFilteredMap(segmatch::PointCloud* filtered_map);

  // Get ROS parameters.
  void getParameters();

  // TODO(renaud) : using ros::Time(0) means "use the latest available transform". Might solve your problem in relocalizer?
  bool getTransform(const std::string& first_frame,
                    const std::string& second_frame,
                    tf::StampedTransform* transform_ptr,
                    ros::Time transform_time = ros::Time(0));

  //TODO move to segmatch->loadTargetCloud()
  void loadTargetCloud();

  //TODO segmatch_->publishAllButTargetMap(); ?
  void displayMatches(const segmatch::PairwiseMatches& matches) const;

  // Node handle.
  ros::NodeHandle& nh_;

  // Parameters.
  LaserMapperParams params_;

  // Subscribers.
  ros::Subscriber scan_sub_;
  ros::Subscriber loop_closure_sub_;
  ros::Subscriber tf_sub_;
  
  
  // Publishers.
  ros::Publisher trajectory_pub_;
  ros::Publisher odometry_trajectory_pub_;
  ros::Publisher point_cloud_pub_;
  ros::Publisher local_map_pub_;
  ros::Publisher distant_map_pub_;
  ros::Publisher new_fixed_cloud_pub_;
  tf::TransformBroadcaster tf_broadcaster_;

  tf::StampedTransform world_to_odom_;
  std::mutex world_to_odom_mutex_;

  // Services.
  ros::ServiceServer save_distant_map_;
  ros::ServiceServer save_map_;

  // Transform communication.
  tf::TransformListener tf_listener_;

  // Sliding Window estimator.
  laser_slam::SlidingWindowEstimator swe_;
  std::mutex swe_mutex_;

  // Contains the map which is estimated by the sliding window.
  segmatch::PointCloud local_map_;
  std::mutex local_map_mutex_;

  segmatch::PointCloud local_map_filtered_;
  std::mutex local_map_filtered_mutex_;

  // Contains the map which is distant from sensor and assumed to be fixed.
  // If the robot revisits the same environment, the distant_map_and local_map_ will be one
  // above each other, each with same density.
  segmatch::PointCloud distant_map_;
  std::mutex distant_map_mutex_;

  // Timestamp to be subtracted to each measurement time so that the trajectory starts at time 0.
  laser_slam::Time base_time_ns_ = 0;

  // Indicates whether the base time was set.
  bool base_time_set_ = false;

  // TODO probably remove?
  unsigned int scan_counter_ = 0;

  static constexpr double kPublishMapThreadRateForServiceCall_hz = 10;
  static constexpr double kPublishCovarianceThreadRateForServiceCall_hz = 10;

  // TODO probably remove?
  // Libpointmatcher density filters.
  laser_slam::PointMatcher::DataPointsFilters max_density_filters_;
  laser_slam::PointMatcher::DataPointsFilters maximum_distance_filters_;
  laser_slam::PointMatcher::DataPointsFilters minimum_distance_filters_;

  laser_slam::SE3 last_pose_;
  bool last_pose_set_ = false;

  pcl::VoxelGrid<segmatch::PclPoint> voxel_filter_;

  // SegMatch objects.
  segmatch_ros::SegMatchWorkerParams segmatch_worker_params_;
  segmatch_ros::SegMatchWorker segmatch_worker_;
  static constexpr unsigned int kSegMatchThreadRate = 10u;

  // Indicates whether a new source cloud is ready for localization or loop-closure.
  bool source_cloud_ready_ = false;
  std::mutex source_cloud_ready_mutex_;

  static constexpr double kTimeout_s = 0.2;
  static constexpr unsigned int kScanSubscriberMessageQueueSize = 1u;
  static constexpr unsigned int kPublisherQueueSize = 50u;
}; // LaserMapper

#endif /* LASER_MAPPER_LASER_MAPPER_HPP_ */
