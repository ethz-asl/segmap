#include "segmatch/normal_estimators/simple_normal_estimator.hpp"

namespace segmatch {

SimpleNormalEstimator::SimpleNormalEstimator(const float search_radius) {
  // Ensure that the normals point to the same direction.
  normal_estimator_.setViewPoint(std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max());
  normal_estimator_.setRadiusSearch(search_radius);
}

std::vector<bool> SimpleNormalEstimator::updateNormals(
    const MapCloud& points, const std::vector<int>& points_mapping,
    const std::vector<int>& new_points_indices,
    PointsNeighborsProvider<MapPoint>& points_neighbors_provider) {
  CHECK(points_neighbors_provider.getPclSearchObject() != nullptr) <<
      "SimpleNormalEstimator can only be used with point neighbors providers that expose a PCL "
      "search object.";

  MapCloud::ConstPtr input_cloud(&points, [](MapCloud const* ptr) {});
  normal_estimator_.setSearchMethod(points_neighbors_provider.getPclSearchObject());
  normal_estimator_.setInputCloud(input_cloud);
  normal_estimator_.compute(normals_);

  return std::vector<bool>(points.size(), true);
}

} // namespace segmatch
