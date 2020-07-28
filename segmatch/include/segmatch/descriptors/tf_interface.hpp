#ifndef SEGMATCH_TF_INTERFACE_HPP_
#define SEGMATCH_TF_INTERFACE_HPP_

#include "ros/ros.h"
#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/tensorflow_msg.h"
#include "std_msgs/String.h"
#include "std_msgs/UInt64.h"

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
  TensorflowInterface(ros::NodeHandle& nh);

  void sendMessage(std::string s);

  void batchFullForwardPass(const std::vector<Array3D>& inputs,
                            const std::string& input_tensor_name,
                            const std::vector<std::vector<float> >& scales,
                            const std::string& scales_tensor_name,
                            const std::string& descriptor_tensor_name,
                            const std::string& reconstruction_tensor_name,
                            std::vector<std::vector<float> >& descriptors,
                            std::vector<Array3D>& reconstructions) const;

 private:
  ros::Publisher publisher_batch_full_forward_pass_;
  ros::NodeHandle nh_;
};

}  // namespace tf_interface

#endif  // SEGMATCH_TF_INTERFACE_HPP_
