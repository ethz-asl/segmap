#ifndef SEGMATCH_POINTS_NEIGHBORS_PROVIDER_HPP_
#define SEGMATCH_POINTS_NEIGHBORS_PROVIDER_HPP_

#include <vector>

#include <pcl/search/search.h>

#include "segmatch/common.hpp"

namespace segmatch {

/// \brief Indices of the neighbors of a point.
typedef std::vector<int> PointNeighbors;

/// \brief Provides point neighborhood information of a point cloud.
/// \remark The whole refactoring based on the PointsNeighborsProvider interface was meant
/// especially for the caching of the nearest neighbors. Unfortunately this turned out to be too
/// slow because of index remapping overhead. The interface is kept anyway for future work that
/// will require it (persisting points positions in the point clouds will reduce the overhead and
/// enable caching).
template<typename PointT>
class PointsNeighborsProvider {
 public:
  /// \brief Finalizes an instance of the PointsNeighborsProvider class.
  virtual ~PointsNeighborsProvider() = default;

  /// \brief Update the points neighborhood provider.
  /// \param point_cloud The new point cloud.
  /// \param points_mapping Mapping from the points stored in the current cloud to the points of
  /// the new point cloud. Point \c i is moved to position \c points_mapping[i]. Values smaller
  /// than 0 indicate that the point has been removed.
  /// \remarks point_cloud must remain a valid object during all the successive calls to
  /// getNeighborsOf()
  virtual void update(const typename pcl::PointCloud<PointT>::ConstPtr point_cloud,
                      const std::vector<int64_t>& points_mapping) = 0;

  /// \brief Gets the indexes of the neighbors of the point with the specified index.
  /// \param point_index Index of the query point.
  /// \param search_radius The radius of the searched neighborhood.
  /// \return Vector containing the indices of the neighbor points.
  virtual const PointNeighbors getNeighborsOf(size_t point_index, float search_radius) = 0;

  /// \brief Returns the underlying PCL search object.
  /// \remarks This function is present only for compatibility with the old segmenters and
  /// should not be used in new code.
  /// \returns Pointer to the PCL search object. If the provider doesn't use PCL search objects
  /// this function returns null.
  virtual typename pcl::search::Search<PointT>::Ptr getPclSearchObject() = 0;
}; // class DynamicPointNeighborsProvider

} // namespace segmatch

#endif // SEGMATCH_POINTS_NEIGHBORS_PROVIDER_HPP_
