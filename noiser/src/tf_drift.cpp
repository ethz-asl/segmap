#include "noiser/tf_drift.hpp"

#include <iostream>

namespace noiser {

TfDriftClass::TfDriftClass() {
  ros::NodeHandle nh;
  nh.param<float>("tf_drift/noise_x_mean", noise_x_mean_, 0.0);
  nh.param<float>("tf_drift/noise_x_stddev", noise_x_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_y_mean", noise_y_mean_, 0.0);
  nh.param<float>("tf_drift/noise_y_stddev", noise_y_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_z_mean", noise_z_mean_, 0.0);
  nh.param<float>("tf_drift/noise_z_stddev", noise_z_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_yaw_mean", noise_yaw_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_yaw_stddev", noise_yaw_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_attitude_mean", noise_attitude_mean_, 0.0);
  nh.param<float>("tf_drift/noise_attitude_stddev", noise_attitude_stddev_, 0.01);
  nh.param<std::string>("tf_drift/source_frame", source_frame_, "world_drift");
  nh.param<std::string>("tf_drift/target_frame", target_frame_, "world");
  nh.param<float>("tf_drift/tf_rate", tf_rate_, 10.0);


  std::cout<<"Target Frame: "<<target_frame_<<std::endl;
}

void TfDriftClass::drift() {
  // float test_vel_x = 0.5; // GT odometry.
  // x_gt += (1.0/tf_rate_)*test_vel_x;

  // PDFs.
  std::normal_distribution<float> dist_x(noise_x_mean_, noise_x_stddev_);
  std::normal_distribution<float> dist_y(noise_y_mean_, noise_y_stddev_);
  std::normal_distribution<float> dist_z(noise_z_mean_, noise_z_stddev_);
  std::normal_distribution<float> dist_yaw(noise_yaw_mean_, noise_yaw_stddev_);
  std::normal_distribution<float> dist_attitude(noise_attitude_mean_, noise_attitude_stddev_);

  // Get relative TF to last frame T_af1af2.

  // Compute TF with incremental noise in yaw: T_d*T_af1af2

  //  

  // Sample yaw noise (non-zero mean!). 
  std::cout<<"Hey"<<drift_x_<<" "<<drift_y_<<" "<<drift_z_<<" "<<drift_yaw_<<std::endl;
  drift_x_ +=  dist_x(generator_);
  drift_y_ +=  dist_y(generator_);
  drift_z_ +=  dist_z(generator_);
  drift_yaw_ += dist_yaw(generator_);
  transform_drifted_.setOrigin(tf::Vector3(drift_x_, drift_y_, drift_z_));
  tf::Quaternion q;

  // Non-cumulative disturbances.
  float noise_roll = dist_attitude(generator_);
  float noise_pitch = dist_attitude(generator_);
  std::cout<<"YO "<<dist_x(generator_)<<" "<<dist_y(generator_)<<" "<<dist_z(generator_)<<std::endl;
  q.setRPY(noise_roll, noise_pitch, drift_yaw_);
  transform_drifted_.setRotation(q);
  br_.sendTransform(tf::StampedTransform(transform_drifted_, ros::Time::now(), source_frame_, target_frame_));
}
}  // namespace noiser

int main(int argc, char** argv) {
  ros::init(argc, argv, "TF_Drift");
  ros::NodeHandle nh;
  float tf_rate;
  nh.param<float>("tf_drift/tf_rate", tf_rate, 10.0);
  noiser::TfDriftClass tf_drifter;
  ros::Rate r(tf_rate);
  while (ros::ok())
  {
    tf_drifter.drift();
    ros::spinOnce();
    r.sleep();
  }
  return 0;
}