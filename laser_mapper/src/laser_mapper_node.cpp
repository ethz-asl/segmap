#include <thread>

#include <ros/ros.h>

#include "laser_mapper/laser_mapper.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "LaserMapper");
  ros::NodeHandle node_handle("~");

  LaserMapper mapper(node_handle);

  std::thread publish_map_thread(&LaserMapper::publishMapThread, &mapper);
  std::thread publish_tf_thread(&LaserMapper::publishTfThread, &mapper);
  std::thread segmatch_thread(&LaserMapper::segMatchThread, &mapper);


  try {
    ros::spin();
  }
  catch (const std::exception& e) {
    ROS_ERROR_STREAM("Exception: " << e.what());
    return 1;
  }
  catch (...) {
    ROS_ERROR_STREAM("Unknown Exception");
    return 1;
  }

  publish_map_thread.join();
  publish_tf_thread.join();
  segmatch_thread.join();

  return 0;
}
