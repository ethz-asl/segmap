#ifndef SEGMATCH_INCREMENTAL_NORMAL_ESTIMATOR_HPP_
#define SEGMATCH_INCREMENTAL_NORMAL_ESTIMATOR_HPP_

#define PCL_NO_PRECOMPILE
#include <pcl/search/kdtree.h>

#include "segmatch/normal_estimators/normal_estimator.hpp"

namespace segmatch {

/// \brief Incremental version of least squares normal estimation.
class IncrementalNormalEstimator : public NormalEstimator {
 public:
  /// \brief Initializes a new instance of the IncrementalNorminalEstimator class.
  /// \param search_radius The search radius used for estimating the normals.
  IncrementalNormalEstimator(float search_radius);

  /// \brief Notifies the estimator that points have been transformed.
  /// \param transformation Linear transformation applied to the points.
  void notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) override;

  /// \brief Clear all the normals and the associated information. Equivalent to removing all the
  /// points from the cloud.
  void clear() override;

  /// \brief Updates the normals of the points of a cloud.
  /// \param points The point cloud for which normals have to be computed.
  /// \param points_mapping Mapping from the indices of the points in the old point cloud to the
  /// indices of the points in the new point cloud since the last update.
  /// \param new_points_indices Indices of the points added to the point cloud since the
  /// last update.
  /// \param points_neighbors_provider Object for nearest neighbors searches.
  /// \returns Vector of booleans indicating which normals or curvatures changed after the update.
  std::vector<bool> updateNormals(
      const MapCloud& points, const std::vector<int>& points_mapping,
      const std::vector<int>& new_points_indices,
      PointsNeighborsProvider<MapPoint>& points_neighbors_provider) override;

  /// \brief Gets the current normal vectors of the point cloud.
  /// \returns Cloud containing the normal vectors.
  const PointNormals& getNormals() const {
    return normals_;
  }

 private:
  // Add contributions of the new points to the normals.
  std::vector<bool> scatterNormalContributions(
      const MapCloud& points, const std::vector<int>& new_points_indices,
      PointsNeighborsProvider<MapPoint>& points_neighbors_provider);

  // Recompute the normals of the specified points.
  void recomputeNormals(const MapCloud& points, const std::vector<bool>& needs_recompute);

  PointNormals normals_;
  float search_radius_;
  pcl::search::KdTree<MapPoint>::Ptr kd_tree_;

  // Partial covariance matrix information for incremental estimation.
  // The covariance matrix is computed as
  // C = E[X*X^t] - mu*mu^t = num_points_ * sum_X_Xt_ + num_points_^2 * sum_X_ * sum_X_^t
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> sum_X_Xt_;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> sum_X_;
  std::vector<size_t> num_points_;
}; // class IncrementalNormalEstimator

} // namespace segmatch

#endif // SEGMATCH_INCREMENTAL_NORMAL_ESTIMATOR_HPP_
