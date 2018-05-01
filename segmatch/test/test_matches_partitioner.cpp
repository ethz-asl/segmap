#include <glog/logging.h>
#include <gtest/gtest.h>

#include "segmatch/recognizers/matches_partitioner.hpp"

using namespace segmatch;

struct PartitionData { };

TEST(MatchesPartitionerTest, test_partitioning) {
  // Arrange
  PairwiseMatches matches {
    PairwiseMatch(4, 56,  { 2.0f,  0.0f, 0.0f }, { -1.0f,  0.0f, -1000.0f }, 1.0f), // (1, 1)
    PairwiseMatch(24, 13, { 4.0f, -6.5f, 0.0f }, { -4.0f, -2.5f,     0.0f }, 1.0f), // (0, 0)
    PairwiseMatch(15, 1,  { 2.0f, 43.0f, 0.0f }, { -1.0f,  1.0f,  1000.0f }, 1.0f), // (1, 1)
    PairwiseMatch(35, 15, { 6.0f,  1.0f, 0.0f }, {  1.5f,  1.0f,     0.0f }, 1.0f)  // (1, 2)
  };

  std::vector<std::pair<size_t, size_t>> expected_partition_indices {
      { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 0 }, { 1, 1 }, { 1, 2 }
  };
  std::vector<std::vector<size_t>> expected_partitions { { 1 }, { }, { }, { }, { 0, 2 }, { 3 } };

  // Act
  MatchesGridPartitioning<PartitionData> partitioning =
      MatchesPartitioner::computeGridPartitioning<PartitionData>(matches, 2.0f);

  // Assert
  ASSERT_EQ(3, partitioning.getWidth());
  ASSERT_EQ(2, partitioning.getHeight());
  for (size_t i = 0u; i < 6; ++i) {
    EXPECT_EQ(expected_partitions[i],
              partitioning(expected_partition_indices[i].first,
                           expected_partition_indices[i].second).match_indices);
  }
}
