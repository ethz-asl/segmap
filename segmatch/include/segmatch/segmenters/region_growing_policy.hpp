#ifndef SEGMATCH_REGION_GROWING_POLICY_HPP_
#define SEGMATCH_REGION_GROWING_POLICY_HPP_

#include "segmatch/common.hpp"
#include "segmatch/parameters.hpp"

namespace segmatch {

/// \brief Policy for region growing segmentation based on euclidean distance criteria. Regions
/// grow to include new points that have a minimum euclidean distance from the points of the
/// region less or equal to a maximum threshold.
struct EuclideanDistance {};

/// \brief Policy for region growing segmentation based on smoothness constraints. Regions grow
/// with the euclidean criteria above and two additional constraints. 1) Growing only happens if
/// the angle between the normal of the seed point and the normal of the candidate point is less or
/// equal to a maximum threshold. 2) A point is used as seed only if its curvature is less or equal
/// to a maximum threshold.
struct SmoothnessConstraints {};

//=================================================================================================
//    Primary template for the RegionGrowingPolicy class
//=================================================================================================

/// \brief Helper class for enabling static_assert() in forbidden primary templates.
template <typename T>
struct dependent_false { enum { value = false }; };

/// \brief Defines a set of rules used by the incremental region growing segmenter.
template<typename PolicyName>
class RegionGrowingPolicy {
 public:
  // Ensure that a valid policy name is specified.
  static_assert(dependent_false<PolicyName>::value, "No implementation found for the requested "
                "segmentation policy. Policies must be implemented as template specializations of "
                "the RegionGrowingPolicy class.");

  /// \brief Parameters of the segmentation policy.
  struct PolicyParameters;

  /// \brief Create the parameters for the segmentation policy.
  /// \param params Parameters of the segmenter.
  /// \returns The parameters of the segmentation policy.
  static PolicyParameters createParameters(const SegmenterParameters& params);

  /// \brief Gets the cluster ID of a target point.
  /// \param point The target point.
  /// \returns The cluster ID of the target point.
  template <typename PointT>
  static uint32_t getPointClusterId(const PointT& point) noexcept;

  /// \brief Sets the cluster ID of a target point.
  /// \param point The target point.
  /// \param cluster_id The new cluster ID of the target point.
  template <typename PointT>
  static void setPointClusterId(PointT& point, const uint32_t cluster_id) noexcept;

  /// \brief Rule for determining if a candidate point can be used as seed for starting or
  /// extending a region.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param point_index Index of the candidate point.
  /// \returns True if the point can be used as seed, false otherwise.
  static bool canPointBeSeed(const PolicyParameters& params, const PointNormals& point_normals,
                             int point_index);

  /// \brief Rule for determining if it is possible to grow a region from a seed point to include
  /// a candidate point.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param seed_index Index of the seed point.
  /// \param candidate_index Index of the candidate point.
  /// \returns True if the candidate point must be included in the seed region, false otherwise.
  static bool canGrowToPoint(const PolicyParameters& params, const PointNormals& point_normals,
                             int seed_index, int candidate_index);

  /// \brief Rule for preparing the seed indices.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param first Random access iterator to the beginning of the indices container.
  /// \param last Random access iterator to the end of the indices container.
  template <typename RandomAccessIterator>
  static void prepareSeedIndices(const PointNormals& point_normals, RandomAccessIterator first,
                                 RandomAccessIterator last);
}; // class RegionGrowingPolicy

//=================================================================================================
//    Specialization of the RegionGrowingPolicy template for EuclideanDistance policies
//=================================================================================================

/// \brief Defines a set of rules used by the incremental region growing segmenter.
template<>
class RegionGrowingPolicy<EuclideanDistance> {
 public:
  /// \brief Parameters of the segmentation policy.
  struct PolicyParameters { };

  /// \brief Create the parameters for the segmentation policy.
  /// \param params Parameters of the segmenter.
  /// \returns The parameters of the segmentation policy.
  static PolicyParameters createParameters(const SegmenterParameters& params) {
    return { };
  }

  /// \brief Gets the cluster ID of a target point.
  /// \param point The target point.
  /// \returns The cluster ID of the target point.
  template <typename PointT>
  static uint32_t getPointClusterId(const PointT& point) noexcept {
    static_assert(pcl::traits::has_field<PointT, pcl::fields::ed_cluster_id>::value,
                  "Region growing segmentation with EuclideanDistance policy can be performed "
                  "only on points containing the ed_cluster_id field.");
    return point.ed_cluster_id;
  }

  /// \brief Sets the cluster ID of a target point.
  /// \param point The target point.
  /// \param cluster_id The new cluster ID of the target point.
  template <typename PointT>
  static void setPointClusterId(PointT& point, const uint32_t cluster_id) noexcept {
    static_assert(pcl::traits::has_field<PointT, pcl::fields::ed_cluster_id>::value,
                  "Region growing segmentation with EuclideanDistance policy can be performed "
                  "only on points containing the ed_cluster_id field.");
    point.ed_cluster_id = cluster_id;
  }

  /// \brief Rule for determining if a candidate point can be used as seed for starting or
  /// extending a region.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param point_index Index of the candidate point.
  /// \returns True if the point can be used as seed, false otherwise.
  static bool canPointBeSeed(const PolicyParameters& params, const PointNormals& point_normals,
                             int point_index) {
    return true;
  }

  /// \brief Rule for determining if it is possible to grow a region from a seed point to include
  /// a candidate point.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param seed_index Index of the seed point.
  /// \param candidate_index Index of the candidate point.
  /// \returns True if the candidate point must be included in the seed region, false otherwise.
  static bool canGrowToPoint(const PolicyParameters& params, const PointNormals& point_normals,
                             int seed_index, int candidate_index) {
    return true;
  }

  /// \brief Rule for preparing the seed indices.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param first Random access iterator to the beginning of the indices container.
  /// \param last Random access iterator to the end of the indices container.
  template <typename RandomAccessIterator>
  static void prepareSeedIndices(const PointNormals& point_normals, RandomAccessIterator first,
                                 RandomAccessIterator last) {
  }
}; // class RegionGrowingPolicy<EuclideanDistance>

//=================================================================================================
//    Specialization of the RegionGrowingPolicy template for SmoothnessConstraints policies
//=================================================================================================

/// \brief Defines a set of rules used by the incremental region growing segmenter.
template<>
class RegionGrowingPolicy<SmoothnessConstraints> {
 public:
  /// \brief Parameters of the segmentation policy.
  struct PolicyParameters {
    /// \brief Threshold on the cosine of the angle between normals for growing.
    float cosine_angle_threshold;

    /// \brief Threshold on the curvature of a point for using it as seed.
    float curvature_threshold;
  };

  /// \brief Create the parameters for the segmentation policy.
  /// \param params Parameters of the segmenter.
  /// \returns The parameters of the segmentation policy.
  static PolicyParameters createParameters(const SegmenterParameters& params) {
    return {
      std::cos(params.sc_smoothness_threshold_deg / 180.0f * static_cast<float>(M_PI)),
      params.sc_curvature_threshold
    };
  }

  /// \brief Gets the cluster ID of a target point.
  /// \param point The target point.
  /// \returns The cluster ID of the target point.
  template <typename PointT>
  static uint32_t getPointClusterId(const PointT& point) noexcept {
    static_assert(pcl::traits::has_field<PointT, pcl::fields::sc_cluster_id>::value,
                  "Region growing segmentation with SmoothnessConstraints policy can be performed "
                  "only on points containing the sc_cluster_id field.");
    return point.sc_cluster_id;
  }

  /// \brief Sets the cluster ID of a target point.
  /// \param point The target point.
  /// \param cluster_id The new cluster ID of the target point.
  template <typename PointT>
  static void setPointClusterId(PointT& point, const uint32_t cluster_id) noexcept {
    static_assert(pcl::traits::has_field<PointT, pcl::fields::ed_cluster_id>::value,
                  "Region growing segmentation with SmoothnessConstraints policy can be performed "
                  "only on points containing the sc_cluster_id field.");
    point.sc_cluster_id = cluster_id;
  }

  /// \brief Rule for determining if a candidate point can be used as seed for starting or
  /// extending a region.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param point_index Index of the candidate point.
  /// \returns True if the point can be used as seed, false otherwise.
  static bool canPointBeSeed(const PolicyParameters& params, const PointNormals& point_normals,
                             int point_index) {
    return point_normals[point_index].curvature <= params.curvature_threshold;
  }

  /// \brief Rule for determining if it is possible to grow a region from a seed point to include
  /// a candidate point.
  /// \param params Parameters of the segmentation policy.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param seed_index Index of the seed point.
  /// \param candidate_index Index of the candidate point.
  /// \returns True if the candidate point must be included in the seed region, false otherwise.
  static bool canGrowToPoint(const PolicyParameters& params, const PointNormals& point_normals,
                             const int seed_index, const int candidate_index) {
    pcl::Vector3fMapConst seed_normal = point_normals[seed_index].getNormalVector3fMap();
    pcl::Vector3fMapConst candidate_normal = point_normals[candidate_index].getNormalVector3fMap();

    // For computational efficiency, use a threshold on the dot product instead of the angle.
    const float dot_product = std::abs(candidate_normal.dot(seed_normal));
    return dot_product >= params.cosine_angle_threshold;
  }

  /// \brief Rule for preparing the seed indices.
  /// \param point_normals Normal vectors to the points of the point cloud.
  /// \param first Random access iterator to the beginning of the indices container.
  /// \param last Random access iterator to the end of the indices container.
  template <typename RandomAccessIterator>
  static void prepareSeedIndices(const PointNormals& point_normals, RandomAccessIterator first,
                                 RandomAccessIterator last) {
    // Sort points in increasing curvature order.
    std::sort(first, last, [&](const int i, const int j) {
      return point_normals[i].curvature < point_normals[j].curvature;
    });
  }
}; // class RegionGrowingPolicy<SmoothnessConstraints>

} // namespace segmatch

#endif // SEGMATCH_REGION_GROWING_POLICY_HPP_
