#ifndef SEGMATCH_NORMAL_ESTIMATOR_HPP_
#define SEGMATCH_NORMAL_ESTIMATOR_HPP_

#include <laser_slam/common.hpp>

#include "segmatch/common.hpp"
#include "segmatch/points_neighbors_providers/points_neighbors_provider.hpp"

namespace segmatch {

/// \brief Base class for estimating normals in a point cloud.
class NormalEstimator {
 public:
  /// \brief Finalizes an instance of the NormalEstimator class.
  virtual ~NormalEstimator() = default;

  /// \brief Notifies the estimator that points have been transformed.
  /// \param transformation Linear transformation applied to the points.
  /// \remarks updateNormals() must still be called so that the normal vectors actually reflect the
  /// transformation.
  virtual void notifyPointsTransformed(
      const kindr::minimal::QuatTransformationTemplate<float>& transformation) = 0;

  /// \brief Clear all the normals and the associated information. Equivalent to removing all the
  /// points from the cloud.
  virtual void clear() = 0;

  /// \brief Updates the normals of the points of a cloud.
  /// \param points The point cloud for which normals have to be computed.
  /// \param points_mapping Mapping from the indices of the points in the old point cloud to the
  /// indices of the points in the new point cloud since the last update.
  /// \param new_points_indices Indices of the points added to the point cloud since the
  /// last update.
  /// \param points_neighbors_provider Object for nearest neighbors searches.
  /// \returns Vector of booleans indicating which normals or curvatures changed after the update.
  virtual std::vector<bool> updateNormals(
      const MapCloud& points, const std::vector<int>& points_mapping,
      const std::vector<int>& new_points_indices,
      PointsNeighborsProvider<MapPoint>& points_neighbors_provider) = 0;

  /// \brief Gets the current normal vectors of the point cloud.
  /// \returns Cloud containing the normal vectors.
  virtual const PointNormals& getNormals() const = 0;

  /// \brief Creates a normal estimator with the passed parameters.
  /// \param estimator_type Type of estimator. Can be "simple" or "incremental".
  /// \param radius_for_estimation_m The search radius used for the estimation of the normals.
  static std::unique_ptr<NormalEstimator> create(const std::string& estimator_type,
                                                 float radius_for_estimation_m);
}; // class NormalEstimator

} // namespace segmatch

#endif // SEGMATCH_NORMAL_ESTIMATOR_HPP_
