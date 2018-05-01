#ifndef SEGMATCH_UTILITIES_HPP_
#define SEGMATCH_UTILITIES_HPP_

#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stddef.h>

#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include "segmatch/common.hpp"

namespace segmatch {

typedef kindr::minimal::QuatTransformationTemplate<double> SE3;

template <typename PointCloudT>
static void loadCloud(const std::string& filename, PointCloudT* cloud) {
  LOG(INFO) <<"Loading cloud: " << filename << ".";
  CHECK_NE(pcl::io::loadPCDFile(filename, *cloud), -1) <<
      "Failed to load cloud: " << filename << ".";
}

template <typename PointCloudT>
static void translateCloud(const Translation& translation, PointCloudT* cloud) {
  for (size_t i = 0u; i < cloud->size(); ++i) {
    cloud->points[i].x += translation.x;
    cloud->points[i].y += translation.y;
    cloud->points[i].z += translation.z;
  }
}

static void displayPerformances(unsigned int tp, unsigned int tn,
                                unsigned int fp, unsigned int fn) {

  LOG(INFO) << "TP: " << tp << ", TN: " << tn <<
      ", FP: " << fp << ", FN: " << fn << ".";

  const double true_positive_rate = double(tp) / double(tp + fn);
  const double true_negative_rate = double(tn) / double(fp + tn);
  const double false_positive_rate = 1.0 - true_negative_rate;

  LOG(INFO) << "Accuracy (ACC): " << double(tp + tn) /
      double(tp + fp + tn + fn);
  LOG(INFO) << "Sensitivity (TPR): " << true_positive_rate;
  LOG(INFO) << "Specificity (TNR): " << true_negative_rate;
  LOG(INFO) << "Precision: " << double(tp) / double(tp + fp);
  LOG(INFO) << "Positive likelihood ratio: " << true_positive_rate / false_positive_rate;
}

static SE3 fromApproximateTransformationMatrix(const Eigen::Matrix4f& matrix) {
  SE3::Rotation rot = SE3::Rotation::fromApproximateRotationMatrix(
      matrix.cast<double>().topLeftCorner<3,3>().eval());
  SE3::Position pos = matrix.cast<double>().topRightCorner<3,1>().eval();
  return SE3(rot, pos);
}

static PclPoint se3ToPclPoint(const SE3& transform) {
  PclPoint point;
  point.x = transform.getPosition()(0);
  point.y = transform.getPosition()(1);
  point.z = transform.getPosition()(2);
  return point;
}

static double pointToPointDistance(const PclPoint& p1, const PclPoint& p2) {
  return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) +
              (p1.z-p2.z)*(p1.z-p2.z));
}

template<typename PointCloudT>
static void transformPointCloud(const SE3& transform, PointCloudT* point_cloud) {
  CHECK_NOTNULL(point_cloud);
  const Eigen::Matrix4f transform_matrix = transform.getTransformationMatrix().cast<float>();
  pcl::transformPointCloud(*point_cloud, *point_cloud, transform_matrix);
}

static void transformPclPoint(const SE3& transform, PclPoint* point) {
  CHECK_NOTNULL(point);
  Eigen::Matrix<double, 3, 1> eigen_point;
  eigen_point << point->x, point->y, point->z;
  eigen_point = transform.transform(eigen_point);
  point->x = eigen_point(0);
  point->y = eigen_point(1);
  point->z = eigen_point(2);
}

template<typename T>
static T findMostOccuringElement(const std::vector<T>& elements) {
  CHECK(!elements.empty());
  std::map<T, unsigned int> counts;
  for (const auto& element: elements) {
    counts[element]++;
  }
  unsigned int max_count = 0u;
  T most_occuring_element;
  for (const auto& count: counts) {
    if (count.second > max_count) {
      max_count = count.second;
      most_occuring_element = count.first;
    }
  }
  return most_occuring_element;
}

static void applyRandomFilterToCloud(double ratio_of_points_to_keep,
                                     PointICloud* point_cloud) {
  if (ratio_of_points_to_keep != 1.0) {
    CHECK_NOTNULL(point_cloud);
    PointICloud filtered_cloud;
    // Manual filtering as pcl::RandomSample seems to be broken.
    for (size_t i = 0u; i < point_cloud->size(); ++i) {
      if (double(std::rand()) / double(RAND_MAX) < ratio_of_points_to_keep) {
        filtered_cloud.points.push_back(point_cloud->points[i]);
      }
    }

    filtered_cloud.width = 1;
    filtered_cloud.height = filtered_cloud.points.size();
    LOG(INFO) << "Filtering cloud from " << point_cloud->points.size() <<
        " to " << filtered_cloud.points.size() << " points.";
    *point_cloud = filtered_cloud;
  }
}

static PclPoint calculateCentroid(const PointCloud& point_cloud){
  std::feclearexcept(FE_ALL_EXCEPT);
  // Find the mean position of a segment.
  double x_mean = 0.0;
  double y_mean = 0.0;
  double z_mean = 0.0;
  const size_t n_points = point_cloud.points.size();
  for (const auto& point : point_cloud.points) {
    x_mean += point.x / n_points;
    y_mean += point.y / n_points;
    z_mean += point.z / n_points;
  }

  // Check that there were no overflows, underflows, or invalid float operations.
  if (std::fetestexcept(FE_OVERFLOW)) {
    LOG(ERROR) << "Overflow error in centroid computation.";
  } else if (std::fetestexcept(FE_UNDERFLOW)) {
    LOG(ERROR) << "Underflow error in centroid computation.";
  } else if (std::fetestexcept(FE_INVALID)) {
    LOG(ERROR) << "Invalid Flag error in centroid computation.";
  } else if (std::fetestexcept(FE_DIVBYZERO)) {
    LOG(ERROR) << "Divide by zero error in centroid computation.";
  }

  return PclPoint(x_mean, y_mean, z_mean);
}

static PointCloud mapPoint2PointCloud(const MapCloud& map_cloud) {
    PointCloud point_cloud;
    for (const auto& point : map_cloud.points) {
        PclPoint pcl_point;
        pcl_point.x = point.x;
        pcl_point.y = point.y;
        pcl_point.z = point.z;
        point_cloud.points.push_back(pcl_point);
    }
    point_cloud.width = 1;
    point_cloud.height = point_cloud.points.size();
    return point_cloud;
}

} // namespace segmatch

#endif // SEGMATCH_UTILITIES_HPP_
