#include "noiser/tf_drift.hpp"

#include <fstream>
#include <iostream>

namespace noiser {

TfDriftClass::TfDriftClass() {
  ros::NodeHandle nh;
  nh.param<bool>("tf_drift/enable_drift", enable_drift_, false);
  nh.param<float>("tf_drift/noise_x_mean", noise_x_mean_, 0.0);
  nh.param<float>("tf_drift/noise_x_stddev", noise_x_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_y_mean", noise_y_mean_, 0.0);
  nh.param<float>("tf_drift/noise_y_stddev", noise_y_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_z_mean", noise_z_mean_, 0.0);
  nh.param<float>("tf_drift/noise_z_stddev", noise_z_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_yaw_mean", noise_yaw_mean_, 0.1);
  nh.param<float>("tf_drift/noise_yaw_stddev", noise_yaw_stddev_, 0.1);
  nh.param<float>("tf_drift/noise_attitude_mean", noise_attitude_mean_, 0.0);
  nh.param<float>("tf_drift/noise_attitude_stddev", noise_attitude_stddev_, 0.01);
  nh.param<std::string>("tf_drift/odom_drift_frame", odom_drift_frame_, "world_drift");
  nh.param<std::string>("tf_drift/odom_frame", odom_frame_, "world");
  nh.param<std::string>("tf_drift/baselink_frame", baselink_frame_, "base_link");
  nh.param<float>("tf_drift/tf_rate", tf_rate_, 10.0);

  // Init.
  T_W_BdLast_.setIdentity();
  T_W_BLast_.setIdentity();
  dist_x_.param(std::normal_distribution<float>::param_type(noise_x_mean_, noise_x_stddev_));
  dist_y_.param(std::normal_distribution<float>::param_type(noise_y_mean_, noise_y_stddev_));
  dist_z_.param(std::normal_distribution<float>::param_type(noise_z_mean_, noise_z_stddev_));
  dist_yaw_.param(std::normal_distribution<float>::param_type(noise_yaw_mean_, noise_yaw_stddev_));
  dist_attitude_.param(std::normal_distribution<float>::param_type(noise_attitude_mean_, noise_attitude_stddev_));

  export_drift_srv_ = nh.advertiseService(
    "export_drift_transform",
    &TfDriftClass::exportDriftValuesServiceCall, this);

}

void TfDriftClass::driftReal()
{
  tf::Transform T_Wd_W;
  std::string stamp;
  if(enable_drift_)
  {
    // T_ = Tf, W = GT Odom Frame, B = GT Base Link, B* = Drifted Base Link, W* = Drifted Odom Frame, W', B', W*', B*' = Previous timestep.
  
    // Get T_WB from TF tree.
    tf::StampedTransform TS_W_B;
    try{
      listener_.lookupTransform(odom_frame_, baselink_frame_, ros::Time(0), TS_W_B);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      // ToDo: How to handle? Just don't add noise?
      return;
    }

    // ToDo Check if enough time has passed since last update.
    tf::Transform T_W_B = TS_W_B;

    // Compute T_B'B (GT relative motion since last step).
    tf::Transform T_BLast_B = (T_W_BLast_.inverse())*T_W_B; 
    tf::Vector3 transl_gt = T_BLast_B.getOrigin();
    tf::Quaternion quat_gt = T_BLast_B.getRotation();

    float travelled_distance = transl_gt.length();

    // Compute T_B*'B* = addNoise(T_B'B). Noisy relative motion since last step.
    // Sample noise.
    // std::normal_distribution<float> dist_x(noise_x_mean_, noise_x_stddev_);
    float noise_x = dist_x_(generator_);
    float noise_y = dist_y_(generator_);
    float noise_z = dist_z_(generator_);
    float noise_yaw = dist_yaw_(generator_);
    float noise_roll = dist_attitude_(generator_);
    float noise_pitch = dist_attitude_(generator_);
    tf::Quaternion quat_noise;
    quat_noise.setRPY(noise_roll, noise_pitch, noise_yaw);

    transl_gt.setX(noise_x + T_BLast_B.getOrigin().x());
    transl_gt.setY(noise_y + T_BLast_B.getOrigin().y());
    transl_gt.setZ(noise_z + T_BLast_B.getOrigin().z());
    tf::Transform T_BdLast_Bd = T_BLast_B; // ToDo!! Add the magic here.
    T_BdLast_Bd.setOrigin(transl_gt);
    T_BdLast_Bd.setRotation(quat_gt*quat_noise);

    
    // Compute T_WB* = T_WB*' * T_B*'B* (the current drifted base link).
    tf::Transform T_W_Bd = T_W_BdLast_*T_BdLast_Bd;
    // std::cout<<"Hallo5a "<<T_W_BdLast_.getOrigin().x()<<" "<<T_W_BdLast_.getOrigin().y()<<" "<<T_W_BdLast_.getOrigin().z()<<" "<<T_W_BdLast_.getRotation().x()<<" "<<T_W_BdLast_.getRotation().y()<<" "<<T_W_BdLast_.getRotation().z()<<" "<<T_W_BdLast_.getRotation().w()<<std::endl;
    
    // Swap of drifting variable: T_W*B = T_WB* (keep B, add W* instead).
    tf::Transform T_WdB = T_W_Bd;
    
    // Compute T_W*W = T_WB* * inv(T_WB)
    T_Wd_W = T_WdB*(T_W_B.inverse());
    
    // Broadcast T_W*W to TF tree.
    br_.sendTransform(tf::StampedTransform(T_Wd_W, TS_W_B.stamp_, odom_drift_frame_, odom_frame_));
    stamp = std::to_string(TS_W_B.stamp_.toSec());
    
    // Store T_WB*' = T_WB*
    T_W_BdLast_ = T_W_Bd;
    
    // Store T_WB' = T_WB
    T_W_BLast_ = T_W_B;
  }
  else
  {
    T_Wd_W.setIdentity();
    br_.sendTransform(tf::StampedTransform(T_Wd_W, ros::Time::now(), odom_drift_frame_, odom_frame_));
    stamp = std::to_string(ros::Time::now().toSec());
  }

  // Store T_Wd_W.
  // std::cout<<"Timestamp: "<<std::to_string(TS_W_B.stamp_.toSec())<<std::endl;
  std::vector<float> T_Wd_W_vec = {
  T_Wd_W.getOrigin().x(),
  T_Wd_W.getOrigin().y(),
  T_Wd_W.getOrigin().z(),
  T_Wd_W.getRotation().x(),
  T_Wd_W.getRotation().y(),
  T_Wd_W.getRotation().z(),
  T_Wd_W.getRotation().w()
  };
  T_Wd_W_vec_.push_back(T_Wd_W_vec);
  T_Wd_W_stamp_vec_.push_back(stamp);
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
  drift_x_ +=  dist_x(generator_);
  drift_y_ +=  dist_y(generator_);
  drift_z_ +=  dist_z(generator_);
  drift_yaw_ += dist_yaw(generator_);
  transform_drifted_.setOrigin(tf::Vector3(drift_x_, drift_y_, drift_z_));
  tf::Quaternion q;

  // Non-cumulative disturbances.
  float noise_roll = dist_attitude(generator_);
  float noise_pitch = dist_attitude(generator_);
  q.setRPY(noise_roll, noise_pitch, drift_yaw_);
  transform_drifted_.setRotation(q);
  br_.sendTransform(tf::StampedTransform(transform_drifted_, ros::Time::now(), odom_drift_frame_, odom_frame_));
}

bool TfDriftClass::exportDriftValuesServiceCall(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
  std::ofstream output_file;
  output_file.open("/tmp/online_matcher/drift.csv", std::ofstream::out | std::ofstream::trunc);
  output_file << "odom_drift_frame: "<<odom_drift_frame_<<" odom_frame: "<<odom_frame_<<" baselink_frame: "<<baselink_frame_<<std::endl;
  int i = 0;
  for(auto it = T_Wd_W_vec_.begin();it != T_Wd_W_vec_.end(); it++)
  {
    output_file << T_Wd_W_stamp_vec_[i] << " "; // Timestamp.
    output_file << it->at(0) << " "; // t_x.
    output_file << it->at(1) << " "; // t_y.
    output_file << it->at(2) << " "; // t_z.
    output_file << it->at(3) << " "; // q_x.
    output_file << it->at(4) << " "; // q_y.
    output_file << it->at(5) << " "; // q_z.
    output_file << it->at(6) << " "; // q_w.
    output_file << std::endl;
    i++;
  }

  output_file.close();
  return true;
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
    tf_drifter.driftReal();
    ros::spinOnce();
    r.sleep();
  }
  return 0;
}