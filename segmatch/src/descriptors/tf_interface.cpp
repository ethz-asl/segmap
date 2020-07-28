#include "segmatch/descriptors/tf_interface.hpp"

namespace ns_tf_interface {

TensorflowInterface::TensorflowInterface(ros::NodeHandle& nh) : nh_(nh) {
  publisher_batch_full_forward_pass_ = nh_.advertise<segmatch::tensorflow_msg>(
      "tf_interface_topic/tensorflow_msg", 50u);
  ROS_INFO_STREAM("advertising");
  ros::Rate loop_rate(10);
  ros::spinOnce();
  loop_rate.sleep();
}
void TensorflowInterface::sendMessage(std::string s) {
  segmatch::tensorflow_msg msg;

  msg.data = s;
  msg.timestamp = ros::Time::now().toNSec();
  ROS_INFO_STREAM("Sending at: " << msg.timestamp);
  publisher_batch_full_forward_pass_.publish(msg);
  ros::Rate loop_rate(10);
  ros::spinOnce();
  loop_rate.sleep();
}

void TensorflowInterface::batchFullForwardPass(
    const std::vector<Array3D>& inputs, const std::string& input_tensor_name,
    const std::vector<std::vector<float> >& scales,
    const std::string& scales_tensor_name,
    const std::string& descriptor_values_name,
    const std::string& reconstruction_values_name,
    std::vector<std::vector<float> >& descriptors,
    std::vector<Array3D>& reconstructions) const {
  CHECK(!inputs.empty());
  descriptors.clear();
  reconstructions.clear();

  segmatch::tensorflow_msg msg;
  std::string s = "Hello";
  msg.data = s;
  msg.timestamp = ros::Time::now().toNSec();
  ROS_INFO_STREAM("sending at: " << msg.timestamp);
  publisher_batch_full_forward_pass_.publish(msg);
  ros::Rate loop_rate(10);
  ros::spinOnce();
  loop_rate.sleep();
}

}  // namespace ns_tf_interface