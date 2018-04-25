#include "segmatch/descriptors/ensemble_shape_functions.hpp"

#include <glog/logging.h>

namespace segmatch {

void EnsembleShapeFunctions::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  // Object for storing the ESF descriptor.
  pcl::PointCloud<pcl::ESFSignature640>::Ptr signature(new pcl::PointCloud<pcl::ESFSignature640>);
  PointCloudPtr cloud(new PointCloud);

  pcl::copyPointCloud(segment.getLastView().point_cloud, *cloud);

  esf_estimator_.setInputCloud(cloud);
  esf_estimator_.compute(*signature);

  // After estimating the ensemble of shape functions, the signature should be of size 1.
  CHECK_EQ(signature->size(), 1u);

  Feature feature("ensemble_shape");
  for (unsigned int i = 0u; i < kSignatureDimension; ++i) {
    feature.push_back(FeatureValue("esf_" + std::to_string(i), signature->points[0].histogram[i]));
  }
  CHECK_EQ(feature.size(), kSignatureDimension) << "Feature has the wrong dimension";
  features->replaceByName(feature);
}

} // namespace segmatch
