#include "segmatch/descriptors/cnn_ros_publisher.hpp"

namespace segmatch {

CNNPublisher::CNNPublisher(ros::NodeHandle& nh) : nh_(nh) {
  // publisher_ = nh_.advertise<std_msgs::String>("tensorflow_interface", 50u);
  publisher_ = nh_.advertise<segmatch::tensorflow_msg>("tensorflow_interface", 50u);
}

void CNNPublisher::sendMessage(std::string s) {
  segmatch::tensorflow_msg msg;
  // std_msgs::String msg;
  msg.data = s;
  msg.timestamp = ros::Time::now().toNSec();
  publisher_.publish(msg);
  ROS_INFO_STREAM("sending at: %s" << msg.timestamp);
}

}  // namespace segmatch