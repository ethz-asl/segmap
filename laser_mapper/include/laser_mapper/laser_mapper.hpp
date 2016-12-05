#ifndef LASER_MAPPER_LASER_MAPPER_HPP_
#define LASER_MAPPER_LASER_MAPPER_HPP_

#include <string>
#include <vector>

#include <laser_slam/parameters.hpp>
#include <laser_slam/incremental_estimator.hpp>
#include <laser_slam_ros/laser_slam_worker.hpp>
#include <segmatch/common.hpp>
#include <segmatch_ros/common.hpp>
#include <segmatch_ros/segmatch_worker.hpp>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>

#include "laser_mapper/SaveMap.h"

struct LaserMapperParams {
  bool clear_local_map_after_loop_closure = true;

  // Enable publishing a tf transform from world to odom.
  bool publish_world_to_odom;
  std::string world_frame;
  double tf_publication_rate_hz;

  // Trajectory estimator parameters.
  laser_slam::EstimatorParams online_estimator_params;
}; // struct LaserMapperParams

class LaserMapper {

 public:
  explicit LaserMapper(ros::NodeHandle& n);
  ~LaserMapper();

  /// \brief A thread function for handling map publishing.
  void publishMapThread();

  /// \brief A thread function for updating the transform between world and odom.
  void publishTfThread();

  /// \brief A thread function for localizing and closing loops with SegMatch.
  void segMatchThread();

 protected:
  /// \brief Call back of the save_map service.
  bool saveMapServiceCall(laser_mapper::SaveMap::Request& request,
                          laser_mapper::SaveMap::Response& response);

 private:
  // Get ROS parameters.
  void getParameters();

  // Node handle.
  ros::NodeHandle& nh_;

  // Parameters.
  LaserMapperParams params_;

  tf::TransformBroadcaster tf_broadcaster_;

  // Services.
  ros::ServiceServer save_distant_map_;
  ros::ServiceServer save_map_;

  // Incremental estimator.
  std::shared_ptr<laser_slam::IncrementalEstimator> incremental_estimator_;

  // SegMatch objects.
  segmatch_ros::SegMatchWorkerParams segmatch_worker_params_;
  segmatch_ros::SegMatchWorker segmatch_worker_;
  static constexpr double kSegMatchThreadRate_hz = 3.0;

  unsigned int next_track_id_ = 0u;

  // laser_slam objects.
  std::unique_ptr<laser_slam_ros::LaserSlamWorker> laser_slam_worker_;
  laser_slam_ros::LaserSlamWorkerParams laser_slam_worker_params_;
}; // LaserMapper

#endif /* LASER_MAPPER_LASER_MAPPER_HPP_ */
