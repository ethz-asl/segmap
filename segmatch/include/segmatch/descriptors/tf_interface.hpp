#ifndef SEGMATCH_TF_INTERFACE_HPP_
#define SEGMATCH_TF_INTERFACE_HPP_

#include "ros/ros.h"
#include "segmatch/cnn_input_msg.h"
#include "segmatch/cnn_output_msg.h"
#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/sem_input_msg.h"
#include "segmatch/sem_output_msg.h"
#include "std_msgs/MultiArrayDimension.h"

namespace ns_tf_interface {

struct Array3D {
  Array3D(unsigned int x_dim, unsigned int y_dim, unsigned int z_dim) {
    resize(x_dim, y_dim, z_dim);
    init();
  }

  void resize(unsigned int x_dim, unsigned int y_dim, unsigned int z_dim) {
    container.resize(x_dim);
    for (auto& y : container) {
      y.resize(y_dim);
      for (auto& z : y) {
        z.resize(z_dim);
      }
    }
  }

  void init() {
    for (auto& x : container) {
      for (auto& y : x) {
        for (auto& z : y) {
          z = 0.0;
        }
      }
    }
  }

  std::vector<std::vector<std::vector<double> > > container;
};

class TensorflowInterface {
 public:
  TensorflowInterface();

  void batchFullForwardPass(const std::vector<Array3D>& inputs,
                            const std::string& input_tensor_name,
                            const std::vector<std::vector<float> >& scales,
                            const std::string& scales_tensor_name,
                            const std::string& descriptor_tensor_name,
                            const std::string& reconstruction_tensor_name,
                            std::vector<std::vector<float> >& descriptors,
                            std::vector<Array3D>& reconstructions);

  std::vector<std::vector<float> > batchExecuteGraph(
      const std::vector<std::vector<float> >& inputs,
      const std::string& input_tensor_name,
      const std::string& output_tensor_name);

 private:
  void cnn_output_callback(segmatch::cnn_output_msg msg);
  void sem_output_callback(segmatch::sem_output_msg msg);

  ros::Subscriber cnn_output_subscriber_;
  ros::Subscriber sem_output_subscriber_;
  ros::Publisher cnn_input_publisher_;
  ros::Publisher sem_input_publisher_;

  std::map<uid_t, segmatch::cnn_output_msg> returned_cnn_msgs_;
  std::map<uid_t, segmatch::sem_output_msg> returned_sem_msgs_;
};

}  // namespace ns_tf_interface

#endif  // SEGMATCH_TF_INTERFACE_HPP_
