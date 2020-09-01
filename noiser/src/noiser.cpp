#include "noiser/noiser.hpp"

#include <ctime>
#include <stack>

namespace noiser {

std::stack<clock_t> tictoc_stack;
void tic() { tictoc_stack.push(clock()); }

void toc() {
  std::cout << "Time elapsed: "
            << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
            << std::endl;
  tictoc_stack.pop();
}

NoiserClass::NoiserClass() {
  noise_mean_ = 0.0;
  noise_stddev_ = 0.1;
  std::normal_distribution<float> dist_(noise_mean_, noise_stddev_);

  ros::NodeHandle nh;
  ros::Subscriber sub =
      nh.subscribe("/augmented_cloud", 10, &NoiserClass::pclCallback, this);
  pub_ = nh.advertise<sensor_msgs::PointCloud2>("/noisy_cloud", 10);
  ros::spin();
}

void NoiserClass::pclCallback(
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
  tic();
  std::normal_distribution<float> dist(noise_mean_, noise_stddev_);

  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
  for (auto it = cloud->begin(); it != cloud->end(); ++it) {
    it->x += dist(generator_);
    it->y += dist(generator_);
    it->z += dist(generator_);
  }
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*cloud, output);
  output.header = cloud_msg->header;
  pub_.publish(output);
  toc();
}
}  // namespace noiser

int main(int argc, char** argv) {
  ros::init(argc, argv, "Noiser");
  noiser::NoiserClass noiser_worker;
  return 0;
}