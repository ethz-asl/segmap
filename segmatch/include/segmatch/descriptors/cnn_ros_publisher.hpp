#ifndef SEGMATCH_CNN_ROS_PUBLISHER_HPP_
#define SEGMATCH_CNN_ROS_PUBLISHER_HPP_

#include "ros/ros.h"
#include "segmatch/descriptors/descriptors.hpp"
#include "std_msgs/String.h"
#include "std_msgs/UInt64.h"
#include "segmatch/tensorflow_msg.h"

namespace segmatch {

class CNNPublisher {
 public:
  CNNPublisher(ros::NodeHandle& nh);

  void sendMessage(std::string s);

 private:
  ros::Publisher publisher_;
  ros::NodeHandle nh_;
};

}  // namespace segmatch

#endif  // SEGMATCH_CNN_ROS_PUBLISHER_HPP_
