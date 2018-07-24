#ifndef SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_
#define SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

namespace segmatch {

/// \brief A grid of cubic volume cells.
///
/// Points inserted in the grid are assigned to their respective voxels. The grid provides a
/// downsampled view of the points and supports removing of voxels according to a predicate.
/// The grid distinguishes between <em>active voxels</em> (voxels that contain a minimum number of
/// points) and \e inactive voxels.
/// \remark The class is \e not thread-safe. Concurrent access to the class results in undefined
/// behavior.
template<
  typename InputPointT,
  typename VoxelPointT,
  typename IndexT = uint64_t,
  uint8_t bits_x = 20,
  uint8_t bits_y = 20,
  uint8_t bits_z = 20>
class DynamicVoxelGrid {
 public:
  typedef typename pcl::PointCloud<InputPointT> InputCloud;
  typedef typename pcl::PointCloud<VoxelPointT> VoxelCloud;

  static_assert(std::is_integral<IndexT>::value && std::is_unsigned<IndexT>::value,
                "IndexT must be an unsigned integral type");
  static_assert(bits_x + bits_y + bits_z <= sizeof(IndexT) * 8,
                "The number of bits required per dimension is bigger than the size of IndexT");
  static_assert(bits_x > 0 && bits_y > 0 && bits_z > 0,
                "The index requires at least one bit per dimension");
  static_assert(pcl::traits::has_xyz<InputPointT>::value,
                "InputPointT must be a structure containing XYZ coordinates");
  static_assert(pcl::traits::has_xyz<VoxelPointT>::value,
                "VoxelPointT must be a structure containing XYZ coordinates");

  /// \brief Initializes a new instance of the DynamicVoxelGrid class.
  /// \param resolution Edge length of the voxels.
  /// \param min_points_per_voxel Minimum number of points that a voxel must contain in order to be
  /// considered \e active.
  /// \param origin The point around which the grid is centered.
  DynamicVoxelGrid(const float resolution, const int min_points_per_voxel,
                   const InputPointT& origin = InputPointT())
    : resolution_(resolution)
    , min_points_per_voxel_(min_points_per_voxel)
    , origin_(origin)
    , grid_size_(
        resolution * static_cast<float>(n_voxels_x),
        resolution * static_cast<float>(n_voxels_y),
        resolution * static_cast<float>(n_voxels_z))
    , origin_offset_(origin.getVector3fMap())
    , indexing_offset_(grid_size_ / 2.0f - origin_offset_)
    , world_to_grid_(1.0f / resolution)
    , min_corner_(origin_offset_ - grid_size_ / 2.0f)
    , max_corner_(origin_offset_ + grid_size_ / 2.0f)
    , active_centroids_(new VoxelCloud())
    , inactive_centroids_(new VoxelCloud())
    , pose_transformation_()
    , indexing_transformation_() {

    // Validate inputs.
    CHECK_GT(resolution, 0.0f);
    CHECK_GE(min_points_per_voxel, 1);
  }

  /// \brief Move constructor for the DynamicVoxelGrid class.
  /// \param other Object that has to be moved into the new instance.
  DynamicVoxelGrid(DynamicVoxelGrid&& other)
    : resolution_(other.resolution_)
    , min_points_per_voxel_(other.min_points_per_voxel_)
    , origin_(std::move(other.origin_))
    , grid_size_(std::move(other.grid_size_))
    , origin_offset_(std::move(other.origin_offset_))
    , indexing_offset_(std::move(other.indexing_offset_))
    , world_to_grid_(other.world_to_grid_)
    , min_corner_(std::move(other.min_corner_))
    , max_corner_(std::move(other.max_corner_))
    , active_centroids_(std::move(other.active_centroids_))
    , inactive_centroids_(std::move(other.inactive_centroids_))
    , voxels_(std::move(other.voxels_))
    , pose_transformation_(std::move(other.pose_transformation_))
    , indexing_transformation_(std::move(other.indexing_transformation_)) {
  }

  /// \brief Inserts a point cloud in the voxel grid.
  /// Inserting new points updates the X, Y, Z coordinates of the points, but leaves any extra
  /// fields untouched.
  /// \remark Insertion invalidates any reference to the centroids.
  /// \param new_cloud The new points that must be inserted in the grid.
  /// \returns Indices of the centroids of the voxels that have become \e active after the
  /// insertion.
  std::vector<int> insert(const InputCloud& new_cloud);

  /// \brief Result of a removal operation.
  /// \remark Enabled only for predicates of the form: <tt>bool p(const VoxelPointT&)</tt>
  template<typename Func>
  using RemovalResult = typename std::enable_if<
    std::is_convertible<Func, std::function<bool(const VoxelPointT&)>>::value,
    std::vector<bool>>::type;

  /// \brief Removes from the grid a set of voxels satisfying the given predicate.
  /// \remark Removal invalidates any reference to the centroids.
  /// \returns Vector indicating, for each active voxel index, if the centroid has been removed or
  /// not.
  template <typename Func>
  RemovalResult<Func> removeIf(Func predicate);

  /// \brief Compute the index of the voxel containing the specified point.
  template<typename PointXYZ_>
  IndexT getIndexOf(const PointXYZ_& point) const;

  /// \brief Apply a pose transformation to the voxel grid.
  /// \remark Multiple transformations are cumulative.
  /// \param transformation The transformation to be applied to the grid.
  void transform(const kindr::minimal::QuatTransformationTemplate<float>& transformation);

  /// \brief Clears the dynamic voxel grid, removing all the points it contains and resetting the
  /// transformations.
  void clear();

  /// \brief Returns a reference to the centroids of the active voxels.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \returns The centroids of the active voxels.
  inline VoxelCloud& getActiveCentroids() const { return *active_centroids_; }

  /// \brief Returns a reference to the centroids of the inactive voxels.
  /// \remark Modifying the X, Y, Z components of the points in the returned cloud results in
  /// undefined behavior.
  /// \returns The centroids of the inactive voxels.
  inline VoxelCloud& getInactiveCentroids() const { return *inactive_centroids_; }

  /// \brief Dump informations about the voxels contained in the grid.
  void dumpVoxels() const;

 private:
  // A point with its voxel index.
  struct IndexedPoint_ {
    IndexedPoint_(const InputPointT& point, const IndexT& voxel_index)
      : point(point), voxel_index(voxel_index) {
    }

    InputPointT point;
    IndexT voxel_index;
  };
  typedef std::vector<IndexedPoint_> IndexedPoints_;

  // A voxel in the grid.
  struct Voxel_ {
    Voxel_()
      : centroid(nullptr), index(0), num_points(0) {
    }

    Voxel_(VoxelPointT* centroid, const IndexT& index, const uint32_t num_points)
      : centroid(centroid), index(index), num_points(num_points) {
    }

    Voxel_(const Voxel_& other)
      : centroid(other.centroid), index(other.index), num_points(other.num_points) {
    }

    VoxelPointT* centroid;
    IndexT index;
    uint32_t num_points;
  };

  // The data necessary to construct a voxel.
  struct VoxelData_ {
    Voxel_* old_voxel;
    typename IndexedPoints_::iterator points_begin;
    typename IndexedPoints_::iterator points_end;
  };

  // Compute the voxel indices of a point cloud and sort the points in increasing voxel index
  // order.
  IndexedPoints_ indexAndSortPoints_(const InputCloud& points) const;

  // Create a voxel staring from the data about the point it contains and insert it in the voxels
  // and centroids vectors. Returns true if the new points inserted triggered the voxel.
  bool createVoxel_(const IndexT index, const VoxelData_& data, std::vector<Voxel_>& new_voxels,
                    VoxelCloud& new_active_centroids, VoxelCloud& new_inactive_centroids);

  // Removes the centroids at the specified pointers. The pointers must be sorted in increasing
  // order.
  std::vector<bool> removeCentroids_(VoxelCloud& target_cloud, std::vector<VoxelPointT*> to_remove);

  // The centroids of the voxels containing enough points.
  std::unique_ptr<VoxelCloud> active_centroids_;
  std::unique_ptr<VoxelCloud> inactive_centroids_;

  // The voxels in the point cloud.
  std::vector<Voxel_> voxels_;

  // Properties of the grid.
  const float resolution_;
  const int min_points_per_voxel_;
  const InputPointT origin_;

  // Size of the voxel grid.
  static constexpr IndexT n_voxels_x = (IndexT(1) << bits_x);
  static constexpr IndexT n_voxels_y = (IndexT(1) << bits_y);
  static constexpr IndexT n_voxels_z = (IndexT(1) << bits_z);

  // Variables needed for conversion from world coordinates to voxel index.
  const Eigen::Vector3f grid_size_;
  const Eigen::Vector3f origin_offset_;
  const Eigen::Vector3f indexing_offset_;
  const Eigen::Vector3f min_corner_;
  const Eigen::Vector3f max_corner_;
  float world_to_grid_;
  kindr::minimal::QuatTransformationTemplate<float> pose_transformation_;
  kindr::minimal::QuatTransformationTemplate<float> indexing_transformation_;
}; // class DynamicVoxelGrid

// Short name macros for Dynamic Voxel Grid (DVG) template declaration and
// specification.
#define _DVG_TEMPLATE_DECL_ typename InputPointT, typename VoxelPointT, typename IndexT, uint8_t \
  bits_x, uint8_t bits_y, uint8_t bits_z
#define _DVG_TEMPLATE_SPEC_ InputPointT, VoxelPointT, IndexT, bits_x, bits_y, bits_z

//=================================================================================================
//    DynamicVoxelGrid public methods implementation
//=================================================================================================

template<_DVG_TEMPLATE_DECL_>
template <typename Func>
inline DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::RemovalResult<Func>
DynamicVoxelGrid<_DVG_TEMPLATE_SPEC_>::removeIf(Func predicate) {
  // Setup iterators
  auto v_read = voxels_.begin();
  const auto v_end = voxels_.end();

  // Returns a reference to the point cloud containing the centroid of the
  // specified voxel.
  std::vector<VoxelPointT*> active_centroids_to_remove;
  std::vector<VoxelPointT*> inactive_centroids_to_remove;
  auto get_centroids_container_for = [&](const Voxel_& voxel)
      -> std::vector<VoxelPointT*>& {
    if (voxel.num_points >= min_points_per_voxel_) {
      return active_centroids_to_remove;
    } else {
      return inactive_centroids_to_remove;
    }
  };

  // Remove the voxels and collect the pointers of the centroids that must be
  // removed.
  while(v_read != v_end && !predicate(*(v_read->centroid)))
    ++v_read;

  if (v_read == v_end)
    return std::vector<bool>(active_centroids_->size(), false);
  auto v_write = v_read;
  get_centroids_container_for(*v_read).push_back(v_read->centroid);
  ++v_read;

  for(; v_read != v_end; ++v_read) {
    if(!predicate(*(v_read->centroid))) {
      // Copy the centroid, updating the pointer from the voxel.
      *v_write = *v_read;
      v_write->centroid -= get_centroids_container_for(*v_read).size();
      ++v_write;
    } else {
      // Keep track of the voxels that need to be deleted.
      get_centroids_container_for(*v_read).push_back(v_read->centroid);
    }
  }

  voxels_.erase(v_write, v_end);

  // Actually remove centroids
  removeCentroids_(*inactive_centroids_, inactive_centroids_to_remove);
  return removeCentroids_(*active_centroids_, active_centroids_to_remove);
}

} // namespace segmatch

#endif // SEGMATCH_DYNAMIC_VOXEL_GRID_HPP_
