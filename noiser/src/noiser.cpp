#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>

void chatterCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  ROS_INFO("I heard");
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "Noiser");
  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("/augmented_cloud", 10, chatterCallback);
  ros::spin();

  return 0;
}