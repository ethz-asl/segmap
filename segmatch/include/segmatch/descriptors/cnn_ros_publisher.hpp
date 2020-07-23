#ifndef SEGMATCH_CNN_ROS_PUBLISHER_HPP_
#define SEGMATCH_CNN_ROS_PUBLISHER_HPP_

#include "ros/ros.h"
#include "segmatch/descriptors/descriptors.hpp"
#include "std_msgs/String.h"

namespace segmatch {

class CNNPublisher {
 public:
  CNNPublisher();

  void sendMessage(std::string s);

 private:
   ros::Publisher publisher_;

};

}  // namespace segmatch

#endif  // SEGMATCH_CNN_ROS_PUBLISHER_HPP_
