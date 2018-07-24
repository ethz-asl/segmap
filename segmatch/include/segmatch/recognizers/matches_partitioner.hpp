#ifndef SEGMATCH_MATCHES_PARTITIONER_HPP_
#define SEGMATCH_MATCHES_PARTITIONER_HPP_

#include "segmatch/common.hpp"

namespace segmatch {

/// \brief Describes the grid partitioning of a set of matches.
/// Each partition stores the indices of the elements belonging to it and custom partition data.
template <typename PartitionData>
class MatchesGridPartitioning {
 public:
  /// \brief A partition of the grid. Contains indices of the matches and custom partition data.
  struct Partition {
    /// \brief Indices of the matches belonging to the partition.
    std::vector<size_t> match_indices;
    /// \brief Custom partition data.
    PartitionData data;
  };

  /// \brief Initializes a new instance of the MatchesGridPartitioning class.
  /// \param width Number of partitions on the x axis of the grid.
  /// \param height Number of partitions on the y axis of the grid.
  explicit MatchesGridPartitioning(const size_t width = 0u, const size_t height = 0u)
    : width_(width), height_(height), partitions_(width * height) { }

  /// \brief Gets or sets the data of partition (i, j).
  /// \param i Index of the partition on the y axis.
  /// \param j Index of the partition on the x axis.
  /// \returns Reference to the partition data.
  inline Partition& operator() (const size_t i, const size_t j) {
    return partitions_[i * width_ + j];
  }

  /// \brief Gets the data of partition (i, j).
  /// \param i Index of the partition on the y axis.
  /// \param j Index of the partition on the x axis.
  /// \returns Constant reference to the partition data.
  inline const Partition& operator() (const size_t i, const size_t j) const {
    return partitions_[i * width_ + j];
  }

  /// \brief Gets the number of partitions along the x axis.
  /// \returns The number of partitions along the x axis.
  inline const size_t getWidth() const { return width_; }

  /// \brief Gets the number of partitions along the y axis.
  /// \returns The number of partitions along the y axis.
  inline const size_t getHeight() const { return height_; }

 private:
  // Properties of the grid.
  size_t width_;
  size_t height_;

  // The partition data.
  std::vector<Partition> partitions_;
};

/// \brief Provides helper methods for partitioning matches.
class MatchesPartitioner {
 public:
  /// \brief Prevent initialization of static-only class.
  MatchesPartitioner() = delete;

  /// \brief Partition the given set of matches in a grid of squared subdivisions.
  /// \param matches The matches that need to be partitioned.
  /// \param partition_size Size of one partition of the grid.
  /// \return The computed partitioning. Every partition stores the indices in the passed matches
  /// vector of the matches it contains and custom partition data.
  template <typename PartitionData>
  static MatchesGridPartitioning<PartitionData> computeGridPartitioning(
      const PairwiseMatches& matches, float partition_size);
}; // class MatchesPartitioner

//=================================================================================================
//    MatchesPartitioner public methods implementation
//=================================================================================================

template <typename PartitionData>
MatchesGridPartitioning<PartitionData> MatchesPartitioner::computeGridPartitioning(
    const PairwiseMatches& matches, const float partition_size) {

  // Validate inputs.
  CHECK_GT(partition_size, 0.0f);
  if (matches.empty()) return MatchesGridPartitioning<PartitionData>();

  // Find corners of the partitioning grid.
  Eigen::Vector2f min_corner(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Eigen::Vector2f max_corner(std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
  for (const auto& match : matches) {
    const Eigen::Vector2f target_centroid =
        Eigen::Map<const Eigen::Vector2f>(match.centroids_.second.data);
    min_corner = min_corner.cwiseMin(target_centroid);
    max_corner = max_corner.cwiseMax(target_centroid);
  }

  // Compute grid parameters.
  const float partition_size_inv = 1.0f / partition_size;
  Eigen::Vector2f grid_size = max_corner - min_corner;
  size_t width =
      std::max(static_cast<size_t>(ceil(grid_size.x() * partition_size_inv)), size_t(1u));
  size_t height =
      std::max(static_cast<size_t>(ceil(grid_size.y() * partition_size_inv)), size_t(1u));

  // Assign the matches to their partitions.
  MatchesGridPartitioning<PartitionData> partitioning(width, height);
  for (size_t i = 0; i < matches.size(); ++i) {
    const Eigen::Vector2f& xy_coords =
        Eigen::Map<const Eigen::Vector2f>(matches[i].centroids_.second.data);
    const Eigen::Matrix<size_t, 2, 1> grid_coords =
        ((xy_coords - min_corner) * partition_size_inv).cast<size_t>();
    partitioning(grid_coords.y(), grid_coords.x()).match_indices.push_back(i);
  }

  return partitioning;
}

} // namespace segmatch

#endif // SEGMATCH_MATCHES_PARTITIONER_HPP_
