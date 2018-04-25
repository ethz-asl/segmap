#include <glog/logging.h>
#include <gtest/gtest.h>

#include "recognizer_test_data.hpp"
#include "segmatch/recognizers/partitioned_geometric_consistency_recognizer.hpp"

using namespace segmatch;

// Initialize common objects needed by multiple tests.
class PartitionedGeometricConsistencyRecognizerTest : public ::testing::Test {
 protected:
  PartitionedGeometricConsistencyRecognizer recognizer_;

  PartitionedGeometricConsistencyRecognizerTest()
   : recognizer_(createParameters(1.0, 3), 15.0f) {
  }

  void SetUp() override { }

  void TearDown() override { }

  // Shortcut for setting recognizer related parameters.
  static GeometricConsistencyParams createParameters(double resolution_m, int min_cluster_size) {
    GeometricConsistencyParams parameters;
    parameters.resolution = resolution_m;
    parameters.min_cluster_size = min_cluster_size;
    return parameters;
  }

  static void expectMatchesHaveIdPairs(const PairwiseMatches& matches,
                                       const IdPairs& expected_ids) {
    ASSERT_EQ(matches.size(), expected_ids.size());
    for (const auto& pair : expected_ids) {
      EXPECT_TRUE(std::any_of(matches.begin(), matches.end(),
                              [&](const PairwiseMatch& match) { return match.ids_ == pair; }));
    }
  }

  static void verifyRecognitionResult(const PartitionedGeometricConsistencyRecognizer& recognizer,
                                      const std::vector<IdPairs>& expected_ids) {
    ASSERT_EQ(expected_ids.size(), recognizer.getCandidateClusters().size());
    EXPECT_EQ(expected_ids.size(), recognizer.getCandidateTransformations().size());

    for (size_t i = 0u; i < expected_ids.size(); ++i) {
      expectMatchesHaveIdPairs(recognizer.getCandidateClusters()[i], expected_ids[i]);
    }
  }
};

TEST_F(PartitionedGeometricConsistencyRecognizerTest, test_no_consistency) {
  // Arrange
  PairwiseMatches matches = RecognizerTestData::getMatchesNoCluster();
  std::vector<IdPairs> expected_pairs = RecognizerTestData::getExpectedIdPairsNoCluster();

  // Act
  recognizer_.recognize(matches);

  // Assert
  verifyRecognitionResult(recognizer_, expected_pairs);
}

TEST_F(PartitionedGeometricConsistencyRecognizerTest, test_one_consistent_group) {
  // Arrange
  PairwiseMatches matches = RecognizerTestData::getMatchesOneCluster();
  std::vector<IdPairs> expected_pairs = RecognizerTestData::getExpectedIdPairsOneCluster();

  // Act
  recognizer_.recognize(matches);

  // Assert
  verifyRecognitionResult(recognizer_, expected_pairs);
}

TEST_F(PartitionedGeometricConsistencyRecognizerTest, test_two_consistent_groups) {
  // Arrange
  PairwiseMatches matches = RecognizerTestData::getMatchesTwoClusters();
  // Recognizers based on maximum clique detection only provide one cluster.
  std::vector<IdPairs> expected_pairs = { RecognizerTestData::getExpectedIdPairsTwoClusters()[0] };

  // Act
  recognizer_.recognize(matches);

  // Assert
  verifyRecognitionResult(recognizer_, expected_pairs);
}

TEST_F(PartitionedGeometricConsistencyRecognizerTest, test_pcl_fail) {
  // Arrange
  PairwiseMatches matches = RecognizerTestData::getMatchesPCLFail();
  std::vector<IdPairs> expected_pairs =
      RecognizerTestData::getExpectedPCLSuccess();

  // Act
  recognizer_.recognize(matches);

  // Assert
  verifyRecognitionResult(recognizer_, expected_pairs);
}

TEST_F(PartitionedGeometricConsistencyRecognizerTest, test_pcl_success) {
  // Arrange
  PairwiseMatches matches = RecognizerTestData::getMatchesPCLSuccess();
  std::vector<IdPairs> expected_pairs =
      RecognizerTestData::getExpectedPCLSuccess();

  // Act
  recognizer_.recognize(matches);

  // Assert
  verifyRecognitionResult(recognizer_, expected_pairs);
}
