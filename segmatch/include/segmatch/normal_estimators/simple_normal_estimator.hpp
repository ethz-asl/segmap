#ifndef SEGMATCH_SIMPLE_NORMAL_ESTIMATOR_HPP_
#define SEGMATCH_SIMPLE_NORMAL_ESTIMATOR_HPP_

#define PCL_NO_PRECOMPILE
#include <pcl/features/normal_3d.h>

#include "segmatch/normal_estimators/normal_estimator.hpp"

namespace segmatch {

/// \brief Least squares normal estimation as using the PCL implementation.
class SimpleNormalEstimator : public NormalEstimator {
 public:
  /// \brief Initializes a new instance of the SimpleNorminalEstimator class.
  /// \param search_radius The search radius used for estimating the normals.
  SimpleNormalEstimator(const float search_radius);

  /// \brief Notifies the estimator that points have been transformed.
  /// \param transformation Linear transformation applied to the points.
  void notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) override { }

  /// \brief Clear all the normals and the associated information. Equivalent to removing all the
  /// points from the cloud.
  void clear() override { }

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
  PointNormals normals_;
  pcl::NormalEstimation<MapPoint, PclNormal> normal_estimator_;
}; // class SimpleNormalEstimator

} // namespace segmatch

#endif // SEGMATCH_SIMPLE_NORMAL_ESTIMATOR_HPP_
