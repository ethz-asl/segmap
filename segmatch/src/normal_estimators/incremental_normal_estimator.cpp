#include "segmatch/normal_estimators/incremental_normal_estimator.hpp"

#include <laser_slam/benchmarker.hpp>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>

namespace segmatch {

IncrementalNormalEstimator::IncrementalNormalEstimator(const float search_radius)
  : search_radius_(search_radius), kd_tree_(new pcl::search::KdTree<MapPoint>) {
}

void IncrementalNormalEstimator::notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.NotifyPointsTransformed");
  // Rotate the components of the covariance matrix by rewriting the the points as X := R*X + T
  // where R is the rotation matrix and T the translation vector of the transformation.

  // Get rotation and translation matrix.
  const EIGEN_ALIGN16 Eigen::Matrix3f R = transformation.getRotationMatrix();
  const EIGEN_ALIGN16 Eigen::Vector3f T = transformation.getPosition();
  const EIGEN_ALIGN16 Eigen::Matrix3f Rt = R.transpose();
  const EIGEN_ALIGN16 Eigen::RowVector3f Tt = T.transpose();

  for (size_t i = 0u; i < normals_.size(); ++i) {
    pcl::Vector3fMap normal = normals_[i].getNormalVector3fMap();
    Eigen::Matrix3f& sum_X_Xt = sum_X_Xt_[i];
    Eigen::Vector3f& sum_X = sum_X_[i];
    const float n = num_points_[i];

    // Transform the components of the covariance
    sum_X_Xt = R * sum_X_Xt * Rt + R * sum_X * Tt + T * sum_X.transpose() * Rt + n * T * Tt;
    sum_X = n * T + R * sum_X;
    normal = transformation.getRotation().rotate(normal);
  }
}

void IncrementalNormalEstimator::clear() {
  sum_X_Xt_.clear();
  sum_X_.clear();
  num_points_.clear();
  normals_.clear();
}

// Auxiliary function for rearranging elements in a vector according to a mapping.
template <typename Container, typename Element>
void rearrangeElementsWithMapping(const std::vector<int>& mapping, const size_t new_size,
                                  const Element& default_value, Container& container) {
  Container new_container(new_size, default_value);
  for (size_t i = 0u; i < mapping.size(); ++i) {
    if (mapping[i] >= 0) new_container[mapping[i]] = std::move(container[i]);
  }
  container = std::move(new_container);
}

std::vector<bool> IncrementalNormalEstimator::updateNormals(
    const MapCloud& points, const std::vector<int>& points_mapping,
    const std::vector<int>& new_points_indices,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider) {

  // Rearrange the cached information according to the mapping.
  rearrangeElementsWithMapping(points_mapping, points.size(), Eigen::Matrix3f::Zero(), sum_X_Xt_);
  rearrangeElementsWithMapping(points_mapping, points.size(), Eigen::Vector3f::Zero(), sum_X_);
  rearrangeElementsWithMapping(points_mapping, points.size(), 0u, num_points_);
  rearrangeElementsWithMapping(points_mapping, points.size(), PclNormal(), normals_.points);

  // Scatter the contributions of the new points to the covariance matrices of each point's
  // neighborhood.
  const std::vector<bool> is_normal_affected = scatterNormalContributions(
      points, new_points_indices, points_neighbors_provider);

  // Perform eigenvalues analysis on the affected point's covariances to determine their new
  // normals.
  recomputeNormals(points, is_normal_affected);
  return is_normal_affected;
}

std::vector<bool> IncrementalNormalEstimator::scatterNormalContributions(
    const MapCloud& points, const std::vector<int>& new_points_indices,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.ScatterContributions");

  std::vector<bool> is_new_point(points.size(), false);
  for (auto point_index : new_points_indices) is_new_point[point_index] = true;

  // Scatter information to all the points that are affected by the new points.
  std::vector<bool> is_normal_affected(points.size(), false);
  for (auto point_index : new_points_indices) {
    // Find neighbors.
    std::vector<int> neighbors_indices = points_neighbors_provider.getNeighborsOf(point_index,
                                                                                  search_radius_);
    is_normal_affected[point_index] = true;

    const Eigen::Vector3f& source_point = points[point_index].getVector3fMap();
    for (const auto neighbor_index : neighbors_indices) {
      // Add contribution to the neighbor point.
      is_normal_affected[neighbor_index] = true;
      sum_X_Xt_[neighbor_index] += source_point * source_point.transpose();
      sum_X_[neighbor_index] += source_point;
      ++num_points_[neighbor_index];

      // If the neighbor is an old point, then it also contributes to the normal of the new point.
      if (!is_new_point[neighbor_index]) {
        const Eigen::Vector3f& neighbor_point = points[neighbor_index].getVector3fMap();
        sum_X_Xt_[point_index] += neighbor_point * neighbor_point.transpose();
        sum_X_[point_index] += neighbor_point;
        ++num_points_[point_index];
      }
    }
  }

  return is_normal_affected;
}

void IncrementalNormalEstimator::recomputeNormals(const MapCloud& points,
                                                  const std::vector<bool>& needs_recompute) {
  BENCHMARK_BLOCK("SM.AddNewPoints.EstimateNormals.UpdateNormals");

  CHECK(needs_recompute.size() == sum_X_Xt_.size());
  CHECK(needs_recompute.size() == sum_X_.size());
  CHECK(needs_recompute.size() == num_points_.size());
  CHECK(needs_recompute.size() == normals_.size());

  // Only recompute normals for affected points.
  size_t num_affected_normals = 0u;
  for (size_t i = 0u; i < normals_.size(); ++i) {
    if (needs_recompute[i]) {
      ++num_affected_normals;
      if (num_points_[i] >= 3u) {
        // If there are at least three points in the neighborhood, the normal vector is equal to
        // the eigenvector of the smallest eigenvalue.
        const float norm_factor = 1.0f / static_cast<float>(num_points_[i]);
        const EIGEN_ALIGN16 Eigen::Matrix3f covariance =
            (sum_X_Xt_[i] * norm_factor - sum_X_[i] * norm_factor * sum_X_[i].transpose() * norm_factor);
        pcl::solvePlaneParameters(covariance, normals_[i].normal_x, normals_[i].normal_y,
                                  normals_[i].normal_z, normals_[i].curvature);
        constexpr float view_point_component = std::numeric_limits<float>::max();
        pcl::flipNormalTowardsViewpoint(points[i], view_point_component, view_point_component,
                                        view_point_component, normals_[i].normal_x,
                                        normals_[i].normal_y, normals_[i].normal_z);
      } else {
        // Otherwise we don't have enough data to estimate the normal. Just set it to NaN.
        normals_[i].normal_x = normals_[i].normal_y = normals_[i].normal_z = normals_[i].curvature
            = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.AddNewPoints.EstimateNormals.AffectedNormals", num_affected_normals);
}

} // namespace segmatch
