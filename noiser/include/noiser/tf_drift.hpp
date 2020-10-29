#ifndef TF_DRIFT_HPP_
#define TF_DRIFT_HPP_

#include <iostream>
#include <random>

#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

namespace noiser {
class TfDriftClass {
 public:
  TfDriftClass();

  void drift();
  void driftReal();
  bool exportDriftValuesServiceCall(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

 private:
  float tf_rate_;
  std::string odom_drift_frame_, odom_frame_, baselink_frame_;
  tf::TransformBroadcaster br_;
  tf::TransformListener listener_;
  tf::Transform transform_drifted_; 
  tf::Transform T_W_BLast_;
  tf::Transform T_W_BdLast_;

  float noise_x_mean_, noise_y_mean_, noise_z_mean_, noise_yaw_mean_, noise_attitude_mean_;
  float noise_x_stddev_, noise_y_stddev_, noise_z_stddev_, noise_yaw_stddev_, noise_attitude_stddev_;

  std::default_random_engine generator_;

  std::normal_distribution<float> dist_x_;
  std::normal_distribution<float> dist_y_;
  std::normal_distribution<float> dist_z_;
  std::normal_distribution<float> dist_yaw_;
  std::normal_distribution<float> dist_attitude_;

  std::vector<std::vector<float>> T_Wd_W_vec_;

  float drift_x_ = 0.0, drift_y_ = 0.0, drift_z_ = 0.0;
  float drift_yaw_ = 0.0;

  ros::ServiceServer export_drift_srv_;
};
}  // namespace noiser
#endif  // TF_DRIFT_HPP_