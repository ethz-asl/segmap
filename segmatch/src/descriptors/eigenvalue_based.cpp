#include "segmatch/descriptors/eigenvalue_based.hpp"

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <pcl/common/common.h>

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

// EigenvalueBasedDescriptor methods definition
EigenvalueBasedDescriptor::EigenvalueBasedDescriptor(const DescriptorsParameters& parameters) {}

void EigenvalueBasedDescriptor::describe(const Segment& segment, Features* features) {
  CHECK_NOTNULL(features);
  std::feclearexcept(FE_ALL_EXCEPT);

  // Find the variances.
  const SegmentView& segment_view = segment.getLastView();
  const size_t kNPoints = segment_view.point_cloud.points.size();
  PointCloud variances;
  for (size_t i = 0u; i < kNPoints; ++i) {
    variances.push_back(PclPoint());
    variances.points[i].x = segment_view.point_cloud.points[i].x - segment_view.centroid.x;
    variances.points[i].y = segment_view.point_cloud.points[i].y - segment_view.centroid.y;
    variances.points[i].z = segment_view.point_cloud.points[i].z - segment_view.centroid.z;
  }

  // Find the covariance matrix. Since it is symmetric, we only bother with the upper diagonal.
  const std::vector<size_t> row_indices_to_access = {0,0,0,1,1,2};
  const std::vector<size_t> col_indices_to_access = {0,1,2,1,2,2};
  Eigen::Matrix3f covariance_matrix;
  for (size_t i = 0u; i < row_indices_to_access.size(); ++i) {
    const size_t row = row_indices_to_access[i];
    const size_t col = col_indices_to_access[i];
    double covariance = 0;
    for (size_t k = 0u; k < kNPoints; ++k) {
      covariance += variances.points[k].data[row] * variances.points[k].data[col];
    }
    covariance /= kNPoints;
    covariance_matrix(row,col) = covariance;
    covariance_matrix(col,row) = covariance;
  }

  // Compute eigenvalues of covariance matrix.
  constexpr bool compute_eigenvectors = false;
  Eigen::EigenSolver<Eigen::Matrix3f> eigenvalues_solver(covariance_matrix, compute_eigenvectors);
  std::vector<float> eigenvalues(3, 0.0);
  eigenvalues.at(0) = eigenvalues_solver.eigenvalues()[0].real();
  eigenvalues.at(1) = eigenvalues_solver.eigenvalues()[1].real();
  eigenvalues.at(2) = eigenvalues_solver.eigenvalues()[2].real();
  if (eigenvalues_solver.eigenvalues()[0].imag() != 0.0 ||
      eigenvalues_solver.eigenvalues()[1].imag() != 0.0 ||
      eigenvalues_solver.eigenvalues()[2].imag() != 0.0 ) {
    LOG(ERROR) << "Eigenvalues should not have non-zero imaginary component.";
  }

  // Sort eigenvalues from smallest to largest.
  swap_if_gt(eigenvalues.at(0), eigenvalues.at(1));
  swap_if_gt(eigenvalues.at(0), eigenvalues.at(2));
  swap_if_gt(eigenvalues.at(1), eigenvalues.at(2));

  // Normalize eigenvalues.
  double sum_eigenvalues = eigenvalues.at(0) + eigenvalues.at(1) + eigenvalues.at(2);
  double e1 = eigenvalues.at(0) / sum_eigenvalues;
  double e2 = eigenvalues.at(1) / sum_eigenvalues;
  double e3 = eigenvalues.at(2) / sum_eigenvalues;
  LOG_IF(ERROR, e1 == e2 || e2 == e3 || e1 == e3) << "Eigenvalues should not be equal.";

  // Store inside features.
  const double sum_of_eigenvalues = e1 + e2 + e3;
  constexpr double kOneThird = 1.0/3.0;
  CHECK_NE(e1, 0.0);
  CHECK_NE(sum_of_eigenvalues, 0.0);

  const double kNormalizationPercentile = 1.0;

  const double kLinearityMax = 28890.9 * kNormalizationPercentile;
  const double kPlanarityMax = 95919.2 * kNormalizationPercentile;
  const double kScatteringMax = 124811 * kNormalizationPercentile;
  const double kOmnivarianceMax = 0.278636 * kNormalizationPercentile;
  const double kAnisotropyMax = 124810 * kNormalizationPercentile;
  const double kEigenEntropyMax = 0.956129 * kNormalizationPercentile;
  const double kChangeOfCurvatureMax = 0.99702 * kNormalizationPercentile;

  const double kNPointsMax = 13200 * kNormalizationPercentile;

  Feature eigenvalue_feature("eigenvalue");
  eigenvalue_feature.push_back(FeatureValue("linearity", (e1 - e2) / e1 / kLinearityMax));
  eigenvalue_feature.push_back(FeatureValue("planarity", (e2 - e3) / e1 / kPlanarityMax));
  eigenvalue_feature.push_back(FeatureValue("scattering", e3 / e1 / kScatteringMax));
  eigenvalue_feature.push_back(FeatureValue("omnivariance", std::pow(e1 * e2 * e3, kOneThird) / kOmnivarianceMax));
  eigenvalue_feature.push_back(FeatureValue("anisotropy", (e1 - e3) / e1 / kAnisotropyMax));
  eigenvalue_feature.push_back(FeatureValue("eigen_entropy",
                                            (e1 * std::log(e1)) + (e2 * std::log(e2)) + (e3 * std::log(e3)) / kEigenEntropyMax));
  eigenvalue_feature.push_back(FeatureValue("change_of_curvature", e3 / sum_of_eigenvalues / kChangeOfCurvatureMax));

  PclPoint point_min, point_max;

  pcl::getMinMax3D(segment.getLastView().point_cloud, point_min, point_max);

  double diff_x, diff_y, diff_z;

  diff_x = point_max.x - point_min.x;
  diff_y = point_max.y - point_min.y;
  diff_z = point_max.z - point_min.z;

  if (diff_z < diff_x && diff_z < diff_y) {
    eigenvalue_feature.push_back(FeatureValue("pointing_up", 0.2));
  } else {
    eigenvalue_feature.push_back(FeatureValue("pointing_up", 0.0));
  }

  CHECK_EQ(eigenvalue_feature.size(), kDimension) << "Feature has the wrong dimension";
  features->replaceByName(eigenvalue_feature);

  // Check that there were no overflows, underflows, or invalid float operations.
  if (std::fetestexcept(FE_OVERFLOW)) {
    LOG(ERROR) << "Overflow error in eigenvalue feature computation.";
  } else if (std::fetestexcept(FE_UNDERFLOW)) {
    LOG(ERROR) << "Underflow error in eigenvalue feature computation.";
  } else if (std::fetestexcept(FE_INVALID)) {
    LOG(ERROR) << "Invalid Flag error in eigenvalue feature computation.";
  } else if (std::fetestexcept(FE_DIVBYZERO)) {
    LOG(ERROR) << "Divide by zero error in eigenvalue feature computation.";
  }
}

} // namespace segmatch
