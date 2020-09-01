#include "noiser/noiser.hpp"

namespace noiser {

NoiserClass::NoiserClass() {
  ros::NodeHandle nh;
  nh.getParam("noiser/noise_mean", noise_mean_);
  nh.getParam("noiser/noise_stddev", noise_stddev_);
  nh.getParam("noiser/input_topic_name", input_topic_name_);
  nh.getParam("noiser/output_topic_name", output_topic_name_);
  nh.getParam("noiser/noise_factors", noise_factors_);

  std::normal_distribution<float> dist_(noise_mean_, noise_stddev_);

  ros::Subscriber sub =
      nh.subscribe(input_topic_name_, 10, &NoiserClass::pclCallback, this);

  pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_name_, 10);

  ros::spin();
}

void NoiserClass::pclCallback(
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
  std::normal_distribution<float> dist(noise_mean_, noise_stddev_);

  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
  int semantics_class;
  float noise_factor;
  for (auto it = cloud->begin(); it != cloud->end(); ++it) {
    semantics_class = (it->a) / 7;

    // Adjust for the weird labelling of Bosch
    if (semantics_class > 19) {
      semantics_class -= 11;
    }
    if (semantics_class < noise_factors_.size()) {
      noise_factor = noise_factors_[semantics_class];
    } else {
      noise_factor = 1;
    }
    it->x += noise_factor * dist(generator_);
    it->y += noise_factor * dist(generator_);
    it->z += noise_factor * dist(generator_);
  }
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*cloud, output);
  output.header = cloud_msg->header;
  pub_.publish(output);
}
}  // namespace noiser

int main(int argc, char** argv) {
  ros::init(argc, argv, "Noiser");
  noiser::NoiserClass noiser_worker;
  return 0;
}