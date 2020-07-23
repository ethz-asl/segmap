#include "segmatch/descriptors/cnn_ros_publisher.hpp"

namespace segmatch {

CNNPublisher::CNNPublisher() {
  ros::NodeHandle node_handle("~");
  publisher_ = node_handle.advertise<std_msgs::String>("chatter", 1000);
}

void CNNPublisher::sendMessage(std::string s) {
  std::stringstream ss;
  ss << "hello world " << ros::Time::now();
  std_msgs::String msg;
  msg.data = ss.str();
    LOG(INFO) << ss.str();

  publisher_.publish(msg);
}

}  // namespace segmatch