#ifndef SEGMATCH_LOCAL_MAP_HPP_
#define SEGMATCH_LOCAL_MAP_HPP_

#include <laser_slam/common.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>

#include "segmatch/common.hpp"
#include "segmatch/dynamic_voxel_grid.hpp"
#include "segmatch/normal_estimators/normal_estimator.hpp"
#include "segmatch/points_neighbors_providers/points_neighbors_provider.hpp"

namespace segmatch {

/// \brief Parameters of the local map.
struct LocalMapParameters {
  /// \brief Size of a voxel in the grid.
  float voxel_size_m;
  /// \brief Minimum number of points that a voxel must contain in order to be
  /// considered active.
  int min_points_per_voxel;
  /// \brief Radius of the local map.
  float radius_m;
  /// \brief Minimum vertical distance between a point and the robot.
  float min_vertical_distance_m;
  /// \brief Maximum vertical distance between a point and the robot.
  float max_vertical_distance_m;
  /// \brief Type of the method used for querying nearest neighbors information.
  std::string neighbors_provider_type;
};

/// \brief Manages the local point cloud of a robot. Provides methods for inserting, filtering and
/// segmenting points.
/// \remark The class is \e not thread-safe. Concurrent access to the class results in undefined
/// behavior.
template<typename InputPointT, typename ClusteredPointT>
class LocalMap {
 public:
  typedef DynamicVoxelGrid<InputPointT, ClusteredPointT> VoxelGrid;
  typedef typename VoxelGrid::InputCloud InputCloud;
  typedef typename VoxelGrid::VoxelCloud ClusteredCloud;

  /// \brief Initializes a new instance of the LocalMap class.
  /// \param params The parameters of the local map.
  /// \param normal_estimator Pointer to an object that can be used for estimating the normals. If
  /// null, normals will not be estimated.
  LocalMap(const LocalMapParameters& params, std::unique_ptr<NormalEstimator> normal_estimator);

  /// \brief Move constructor for the LocalMap class.
  /// \param other The object to be moved in this instance.
  LocalMap(LocalMap&& other)
    : voxel_grid_(std::move(other.voxel_grid_))
    , radius_squared_m2_(other.radius_squared_m2_)
    , min_vertical_distance_m_(other.min_vertical_distance_m_)
    , max_vertical_distance_m_(other.max_vertical_distance_m_)
    , points_neighbors_provider_(std::move(other.points_neighbors_provider_))
    , normal_estimator_(std::move(other.normal_estimator_)) {
  };

  /// \brief Update the pose of the robot and add new points to the local map.
  /// \param new_clouds Vector of point clouds to be added.
  /// \param pose The new pose of the robot.
  void updatePoseAndAddPoints(const std::vector<InputCloud>& new_clouds,
                              const laser_slam::Pose& pose);

  /// \brief Apply a pose transformation to the points contained in the local map.
  /// \remark Multiple transformations are cumulative.
  /// \param transformation The transformation to be applied to the local map.
  void transform(const kindr::minimal::QuatTransformationTemplate<float>& transformation);

  /// \brief Clears the local map, removing all the points it contains.
  void clear();

  /// \brief Gets a filtered view of the points contained in the point cloud.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \return Reference to the clustered cloud.
  ClusteredCloud& getFilteredPoints() const {
    return voxel_grid_.getActiveCentroids();
  }

  /// \brief Gets a filtered view of the points contained in the point cloud.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \return Pointer to the clustered cloud.
  typename ClusteredCloud::ConstPtr getFilteredPointsPtr() const {
    return typename ClusteredCloud::ConstPtr(&voxel_grid_.getActiveCentroids(),
                                             [](ClusteredCloud const* ptr) {});
  }

  /// \brief Gets the normals of the points of the local map.
  /// \remark The returned value is valid only if the local map has been constructed with the
  /// \c estimate_normals option set to true. Otherwise, the normal cloud is empty.
  /// \return Normals of the points of the map.
  const PointNormals& getNormals() const {
    if (normal_estimator_ != nullptr)
      return normal_estimator_->getNormals();
    else
      return empty_normals_cloud_;
  }

  /// \brief Gets an object that can be used for nearest neighbors queries on the points of the
  /// local map.
  /// \returns The points neighbors provider object.
  inline PointsNeighborsProvider<ClusteredPointT>& getPointsNeighborsProvider() {
    return *points_neighbors_provider_;
  }

  /// \brief Gets a reference to the current mapping from clusters to segment IDs. Cluster \c i has
  /// segment ID <tt>getClusterToSegmentIdMapping()[i]</tt>.
  /// \return Reference to the mapping between clusters and segment IDs.
  std::vector<Id>& getClusterToSegmentIdMapping() {
    return segment_ids_;
  }

  /// \brief Gets the indices of the normals that have been modified since the last update.
  /// \returns Indices of the modified normals.
  std::vector<bool> getIsNormalModifiedSinceLastUpdate() {
    return is_normal_modified_since_last_update_;
  }

 private:
  std::vector<bool> updatePose(const laser_slam::Pose& pose);
  std::vector<int> addPointsAndGetCreatedVoxels(const std::vector<InputCloud>& new_clouds);
  std::vector<int> buildPointsMapping(const std::vector<bool>& is_point_removed,
                                      const std::vector<int>& new_points_indices);

  VoxelGrid voxel_grid_;

  const float radius_squared_m2_;
  const float min_vertical_distance_m_;
  const float max_vertical_distance_m_;

  std::unique_ptr<PointsNeighborsProvider<ClusteredPointT>> points_neighbors_provider_;
  std::unique_ptr<NormalEstimator> normal_estimator_;
  PointNormals empty_normals_cloud_;

  // Variables needed for working with incremental updates.
  std::vector<Id> segment_ids_;
  std::vector<bool> is_normal_modified_since_last_update_;
}; // class LocalMap

} // namespace segmatch

#endif // SEGMATCH_LOCAL_MAP_HPP_
