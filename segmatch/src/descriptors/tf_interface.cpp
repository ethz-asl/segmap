#include "segmatch/descriptors/tf_interface.hpp"

namespace segmatch {

TensorflowInterface::TensorflowInterface() {
  ros::NodeHandle nh;

  cnn_input_publisher_ = nh.advertise<segmatch::cnn_input_msg>(
      "tf_interface_topic/cnn_input_topic", 50u);
  sem_input_publisher_ = nh.advertise<segmatch::sem_input_msg>(
      "tf_interface_topic/sem_input_topic", 50u);
  cnn_output_subscriber_ =
      nh.subscribe("tf_interface_topic/cnn_output_topic", 1,
                   &TensorflowInterface::cnn_output_callback, this);
  sem_output_subscriber_ =
      nh.subscribe("tf_interface_topic/sem_output_topic", 1,
                   &TensorflowInterface::sem_output_callback, this);
}

void TensorflowInterface::batchFullForwardPass(
    const std::vector<VoxelPointCloud>& inputs, const std::string& input_tensor_name,
    const std::vector<std::vector<float>>& scales,
    const std::string& scales_tensor_name,
    const std::string& descriptor_values_name,
    const std::string& reconstruction_values_name,
    std::vector<std::vector<float>>& descriptors,
    std::vector<PointCloud>& reconstructions) {
  CHECK(!inputs.empty());

  segmatch::cnn_input_msg msg;

  msg.input_tensor_name = input_tensor_name;
  msg.scales_tensor_name = scales_tensor_name;
  msg.descriptor_values_name = descriptor_values_name;
  msg.reconstruction_values_name = reconstruction_values_name;

  std_msgs::MultiArrayDimension layout_dim;

  // Scales
  layout_dim.size = scales.size();
  layout_dim.stride = scales.size() * scales[0].size();
  msg.scales.layout.dim.push_back(layout_dim);

  layout_dim.size = scales[0].size();
  layout_dim.stride = scales[0].size();
  msg.scales.layout.dim.push_back(layout_dim);

  for (int i = 0; i < msg.scales.layout.dim[0].size; ++i) {
    for (int j = 0; j < msg.scales.layout.dim[1].size; ++j) {
      msg.scales.data.push_back(scales[i][j]);
    }
  }

  // Inputs
  layout_dim.size = inputs.size();
  layout_dim.stride = inputs.size();
  msg.input_indexes.layout.dim.push_back(layout_dim);

  size_t total_points = 0;
  for (const VoxelPointCloud& input : inputs) {
    total_points += input.size();
    msg.input_indexes.data.push_back(total_points);
  }

  size_t dim_1 = total_points;
  size_t dim_2 = 7u;

  layout_dim.size = dim_1;
  layout_dim.stride = dim_1 * dim_2;
  msg.inputs.layout.dim.push_back(layout_dim);

  layout_dim.size = dim_2;
  layout_dim.stride = dim_2;
  msg.inputs.layout.dim.push_back(layout_dim);

  for (const VoxelPointCloud& input : inputs) {
    for (const VoxelPoint& point : input) {
      msg.inputs.data.push_back(point.x);
      msg.inputs.data.push_back(point.y);
      msg.inputs.data.push_back(point.z);
      msg.inputs.data.push_back(point.r);
      msg.inputs.data.push_back(point.g);
      msg.inputs.data.push_back(point.b);
      msg.inputs.data.push_back(point.semantic_class);
    }
  }

  auto msg_time_stamp = ros::Time::now().toNSec();
  msg.timestamp = msg_time_stamp;
  ROS_DEBUG_STREAM("Sending CNN Input: " << msg.timestamp);
  cnn_input_publisher_.publish(msg);
  ros::spinOnce();

  ros::Rate wait_rate(10);
  segmatch::cnn_output_msg out_msg;
  while (ros::ok()) {
    ros::spinOnce();
    auto it = returned_cnn_msgs_.find(msg_time_stamp);
    if (it != returned_cnn_msgs_.end()) {
      ROS_DEBUG_STREAM("Found CNN message: " << msg_time_stamp);
      out_msg = it->second;
      returned_cnn_msgs_.erase(it);
      break;
    } else {
      ROS_DEBUG_STREAM("waiting");
      wait_rate.sleep();
      if (ros::Time::now().toNSec() - msg_time_stamp > 10e9) {
        ROS_WARN_STREAM("Delayed CNN message: " << msg_time_stamp);
      }
    }
  }

  // Decoding message
  if (out_msg.descriptors.data.empty()) {
    ROS_WARN_STREAM("No descriptor data");
    return;
  }

  if (out_msg.reconstructions.data.empty()) {
    ROS_WARN_STREAM("No reconstruction data");
    return;
  }

  descriptors.clear();
  reconstructions.clear();

  for (int i = 0; i < out_msg.descriptors.layout.dim[0].size; ++i) {
    std::vector<float> descriptor;
    for (int j = 0; j < out_msg.descriptors.layout.dim[1].size; ++j) {
      descriptor.push_back(out_msg.descriptors.data[
          i * out_msg.descriptors.layout.dim[1].stride + j]);
    }
    descriptors.push_back(descriptor);
  }

  /*for (int i = 0; i < out_msg.reconstructions.layout.dim[0].size; ++i) {
    Array3D reconstruction(out_msg.reconstructions.layout.dim[1].size,
                           out_msg.reconstructions.layout.dim[2].size,
                           out_msg.reconstructions.layout.dim[3].size);
    for (int j = 0; j < out_msg.reconstructions.layout.dim[1].size; ++j) {
      for (int k = 0; k < out_msg.reconstructions.layout.dim[2].size; ++k) {
        for (int l = 0; l < out_msg.reconstructions.layout.dim[3].size; ++l) {
          reconstruction.container[j][k][l] =
              out_msg.reconstructions
                  .data[i * out_msg.reconstructions.layout.dim[1].stride +
                        j * out_msg.reconstructions.layout.dim[2].stride +
                        k * out_msg.reconstructions.layout.dim[3].stride + l];
        }
      }
    }
    reconstructions.push_back(reconstruction);
  }*/
}

void TensorflowInterface::cnn_output_callback(segmatch::cnn_output_msg msg) {
  returned_cnn_msgs_.insert(make_pair(msg.timestamp, msg));
}

void TensorflowInterface::sem_output_callback(segmatch::sem_output_msg msg) {
  returned_sem_msgs_.insert(make_pair(msg.timestamp, msg));
}

std::vector<std::vector<float>> TensorflowInterface::batchExecuteGraph(
    const std::vector<std::vector<float>>& inputs,
    const std::string& input_tensor_name,
    const std::string& output_tensor_name) {
  CHECK(!inputs.empty());
  segmatch::sem_input_msg msg;
  msg.input_tensor_name = input_tensor_name;
  msg.output_tensor_name = output_tensor_name;

  std_msgs::MultiArrayDimension layout_dim;
  layout_dim.size = inputs.size();
  layout_dim.stride = inputs.size() * inputs[0].size();
  msg.inputs.layout.dim.push_back(layout_dim);

  layout_dim.size = inputs[0].size();
  layout_dim.stride = inputs[0].size();
  msg.inputs.layout.dim.push_back(layout_dim);

  for (int i = 0; i < msg.inputs.layout.dim[0].size; ++i) {
    for (int j = 0; j < msg.inputs.layout.dim[1].size; ++j) {
      msg.inputs.data.push_back(inputs[i][j]);
    }
  }

  auto msg_time_stamp = ros::Time::now().toNSec();
  msg.timestamp = msg_time_stamp;
  ROS_DEBUG_STREAM("Sending Semantics Input: " << msg.timestamp);
  sem_input_publisher_.publish(msg);
  ros::spinOnce();

  ros::Rate wait_rate(10);
  segmatch::sem_output_msg out_msg;
  while (ros::ok()) {
    ros::spinOnce();
    auto it = returned_sem_msgs_.find(msg_time_stamp);
    if (it != returned_sem_msgs_.end()) {
      ROS_DEBUG_STREAM("Found Sem message: " << msg_time_stamp);
      out_msg = it->second;
      returned_sem_msgs_.erase(it);
      break;
    } else {
      ROS_DEBUG_STREAM("waiting");
      wait_rate.sleep();
      if (ros::Time::now().toNSec() - msg_time_stamp > 10e9) {
        ROS_WARN_STREAM("Delayed Semantics message: " << msg_time_stamp);
      }
    }
  }

  std::vector<std::vector<float>> semantics;

  if (out_msg.semantics.data.empty()) {
    ROS_WARN_STREAM("No semantics data");
    return semantics;
  }
  for (int i = 0; i < out_msg.semantics.layout.dim[0].size; ++i) {
    std::vector<float> semantic;
    for (int j = 0; j < out_msg.semantics.layout.dim[1].size; ++j) {
      semantic.push_back(
          out_msg.semantics
              .data[i * out_msg.semantics.layout.dim[1].stride + j]);
    }
    semantics.push_back(semantic);
  }

  return semantics;
}

}  // namespace segmatch
