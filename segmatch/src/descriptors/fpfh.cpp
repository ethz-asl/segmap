#include "segmatch/descriptors/fpfh.hpp"

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/common.h>
#include <pcl/features/fpfh.h>
#include <pcl/point_types.h>

#pragma STDC FENV_ACCESS on

namespace segmatch {

/// \brief Utility function for swapping two values.
template<typename T>
bool swap_if_gt(T& a, T& b) {
  if (a > b) {
    std::swap(a, b);
    return true;
  }
  return false;
}

// FpfhDescriptor methods definition
FpfhDescriptor::FpfhDescriptor(const DescriptorsParameters& parameters) {}

void FpfhDescriptor::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  std::feclearexcept(FE_ALL_EXCEPT);

  // Do Stuff in here.

  // Extract point cloud.
  PointCloudPtr cloud(new PointCloud);
  pcl::copyPointCloud(segment.getLastView().point_cloud, *cloud);

  // Extract surface normals for point cloud (how to choose radius?).

  // (How to handle NaN normals?).  

  // Get centroid of segment.

  // Get Z-Axis (= fake normal for centroid makes descriptor invariant to centroid normal)

  // Add centroid at the end of point cloud and surface normal.

  // Create FPFHE class and pass data+normals to it.

  // Create empty kdtree.

  // Create output dataset.

  // Compute largest distance centroid-pt.

  // Set radius-search to allow for all points.

  // Only compute SPFH for centroid.

  // Return.
  std::vector<int> test_fpfh(125, 12); 
  std::generate(test_fpfh.begin(), test_fpfh.end(), std::rand);


  Feature fpfh_feature("fpfh");
  for (size_t j = 0u; j < test_fpfh.size(); ++j) {
      fpfh_feature.push_back(
      FeatureValue("fpfh_" + std::to_string(j), test_fpfh[j]));
  }

  features->replaceByName(fpfh_feature);

}

} // namespace segmatch
