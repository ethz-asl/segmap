#include "segmatch/descriptors/cnn_ros_publisher.hpp"

namespace segmatch {

CNNPublisher::CNNPublisher(ros::NodeHandle& nh) : nh_(nh) {
  publisher_ = nh_.advertise<std_msgs::String>("tensorflow_interface", 50u);
}

void CNNPublisher::sendMessage(std::string s) {
  std_msgs::String msg;
  msg.data = s;
  publisher_.publish(msg);
}

}  // namespace segmatch