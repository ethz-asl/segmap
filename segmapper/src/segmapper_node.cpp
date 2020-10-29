#include "segmapper/segmapper.hpp"

#include <ros/ros.h>
#include <signal.h>
#include <thread>

#include <laser_slam/benchmarker.hpp>

void SigintHandler(int sig) {
  laser_slam::Benchmarker::logStatistics(LOG(INFO));
  ros::shutdown();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "SegMapper", ros::init_options::NoSigintHandler);
  ros::NodeHandle node_handle("~");

  SegMapper mapper(node_handle);

  std::thread publish_map_thread(&SegMapper::publishMapThread, &mapper);
  std::thread publish_tf_thread(&SegMapper::publishTfThread, &mapper);
  std::thread segmatch_thread(&SegMapper::segMatchThread, &mapper);

  signal(SIGINT, SigintHandler);

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
