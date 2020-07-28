#include "segmatch/descriptors/tf_interface.hpp"

namespace ns_tf_interface {

TensorflowInterface::TensorflowInterface() {
  ros::NodeHandle nh;

  publisher_batch_full_forward_pass_ =
      nh.advertise<segmatch::batch_full_forward_pass_msg>(
          "tf_interface_topic/batch_full_forward_pass_topic", 50u);
  ROS_INFO_STREAM("advertising");
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

  segmatch::batch_full_forward_pass_msg msg;

  msg.input_tensor_name = input_tensor_name;
  msg.scales_tensor_name = scales_tensor_name;
  msg.descriptor_values_name = descriptor_values_name;
  msg.reconstruction_values_name = reconstruction_values_name;

  msg.timestamp = ros::Time::now().toNSec();
  ROS_INFO_STREAM("sending at: " << msg.timestamp);
  publisher_batch_full_forward_pass_.publish(msg);
  ros::Rate loop_rate(10);
  ros::spinOnce();
  loop_rate.sleep();
}

}  // namespace ns_tf_interface