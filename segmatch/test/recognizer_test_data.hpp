#ifndef SEGMATCH_RECOGNIZER_TEST_DATA_HPP_
#define SEGMATCH_RECOGNIZER_TEST_DATA_HPP_

#include "segmatch/common.hpp"

using namespace segmatch;

typedef std::vector<IdPair> IdPairs;

/// \brief Common test data for CorrespondenceRecognizer tests.
class RecognizerTestData {
 public:
  /// \brief Prevent instantiation of static class.
  RecognizerTestData() = delete;

  /// \brief Gets test data for a set of matches that doesn't contain any cluster.
  static PairwiseMatches getMatchesNoCluster() {
    return {
      PairwiseMatch(1, 11, { 2.0f, 0.0f, 0.0f }, {   5.0f, 0.0f, 0.0f }, 1.0f), // Group 1
      PairwiseMatch(1, 12, { 2.0f, 0.0f, 0.0f }, {   8.0f, 0.0f, 0.0f }, 1.0f), // Group 2
      PairwiseMatch(2, 13, { 4.0f, 0.0f, 0.0f }, {  10.0f, 0.0f, 0.0f }, 1.0f), // Group 2
      PairwiseMatch(3, 14, { 6.0f, 0.0f, 0.0f }, { 150.0f, 0.0f, 0.0f }, 1.0f)  // Group 3
    };
  }

  /// \brief Gets the expected result for a set of matches that doesn't contain any cluster.
  static std::vector<IdPairs> getExpectedIdPairsNoCluster() {
    return { };
  }

  /// \brief Gets test data for a set of matches that contains one cluster.
  static PairwiseMatches getMatchesOneCluster() {
    return {
      PairwiseMatch(1, 11, {  2.0f, 0.0f, 0.0f }, {   5.0f, 0.0f, 0.0f }, 1.0f),  // Group 1
      PairwiseMatch(1, 12, {  2.0f, 0.0f, 0.0f }, {   8.0f, 0.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(2, 13, {  4.0f, 0.0f, 0.0f }, {  10.0f, 0.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(3, 14, {  6.0f, 0.0f, 0.0f }, { 150.0f, 0.0f, 0.0f }, 1.0f),  // Group 3
      PairwiseMatch(4, 15, {  3.0f, 0.0f, 1.0f }, {   9.0f, 1.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(5, 16, { 16.0f, 0.0f, 0.0f }, { 160.5f, 0.0f, 0.0f }, 1.0f)   // Group 3
    };
  }

  /// \brief Gets the expected result for a set of matches that contains one cluster.
  static std::vector<IdPairs> getExpectedIdPairsOneCluster() {
    return {
      {{ 1, 12 }, { 2, 13 }, { 4, 15 }}
    };
  }

  /// \brief Gets test data for a set of matches that contains two clusters.
  static PairwiseMatches getMatchesTwoClusters() {
    return {
      PairwiseMatch(1, 11, {  2.0f,  0.0f, 0.0f }, {   5.0f, 0.0f,  0.0f }, 1.0f),  // Group 1
      PairwiseMatch(1, 12, {  2.0f,  0.0f, 0.0f }, {   8.0f, 0.0f,  0.0f }, 1.0f),  // Group 2
      PairwiseMatch(2, 13, {  4.0f,  0.0f, 0.0f }, {  10.0f, 0.0f,  0.0f }, 1.0f),  // Group 2
      PairwiseMatch(3, 14, {  6.0f,  0.0f, 0.0f }, { 150.0f, 0.0f,  0.0f }, 1.0f),  // Group 3
      PairwiseMatch(6, 17, {  0.0f, 36.3f, 0.0f }, {   0.0f, 0.0f, 75.0f }, 1.0f),  // Group 4
      PairwiseMatch(4, 15, {  3.0f,  0.0f, 1.0f }, {   9.0f, 1.0f,  0.0f }, 1.0f),  // Group 2
      PairwiseMatch(9, 20, {  2.1f,  0.0f, 0.0f }, {   8.1f, 0.0f,  0.0f }, 1.0f),  // Group 2
      PairwiseMatch(7, 18, {  0.0f, 11.0f, 0.0f }, {   0.0f, 0.0f, 50.0f }, 1.0f),  // Group 4
      PairwiseMatch(5, 16, { 16.0f,  0.0f, 0.0f }, { 160.5f, 0.0f,  0.0f }, 1.0f),  // Group 3
      PairwiseMatch(8, 19, {  0.0f, 31.1f, 0.0f }, {   0.0f, 0.0f, 70.0f }, 1.0f)   // Group 4
    };
  }

  /// \brief Gets the expected result for a set of matches that contains two clusters.
  static std::vector<IdPairs> getExpectedIdPairsTwoClusters() {
    return {
      {{ 1, 12 }, { 2, 13 }, { 4, 15 }, { 9, 20 }},
      {{ 6, 17 }, { 7, 18 }, { 8, 19 }}
    };
  }

  /// \brief Gets test data for a set of matches where the PCL fails detecting the maximum clique.
  static PairwiseMatches getMatchesPCLFail() {
    return {
      PairwiseMatch(1, 11, {  5.0f, -10.0f, 0.0f }, { 55.0f, 60.0f, 0.0f }, 1.0f),  // Group 1
      PairwiseMatch(2, 12, {  0.0f,   0.0f, 0.0f }, { 50.0f, 50.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(5, 15, { 10.0f,   0.0f, 1.0f }, { 60.0f, 50.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(3, 13, {  0.0f,  10.0f, 0.0f }, { 50.0f, 60.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(4, 14, { 10.0f,  10.0f, 0.0f }, { 60.0f, 60.0f, 0.0f }, 1.0f),  // Group 3
    };
  }

  /// \brief Gets the expected result for a set of matches that contains two clusters.
  static std::vector<IdPairs> getExpectedPCLFail() {
    return {
      {{ 1, 11 }, { 2, 12 }, { 5, 15 }}
    };
  }

  /// \brief Gets the previous data set in a different ordering so that the PCL detects the
  /// maximum clique.
  static PairwiseMatches getMatchesPCLSuccess() {
    return {
      PairwiseMatch(3, 13, {  0.0f, 10.0f, 0.0f }, {  50.0f, 60.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(4, 14, {  10.0f, 10.0f, 0.0f }, { 60.0f, 60.0f, 0.0f }, 1.0f),  // Group 3
      PairwiseMatch(2, 12, {  0.0f, 0.0f, 0.0f }, {  50.0f, 50.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(5, 15, {  10.0f, 0.0f, 1.0f }, {  60.0f, 50.0f, 0.0f }, 1.0f),  // Group 2
      PairwiseMatch(1, 11, {  5.0f, -10.0f, 0.0f }, { 55.0f, 60.0f, 0.0f }, 1.0f),  // Group 1
    };
  }

  /// \brief Gets the expected result for a set of matches that contains two clusters.
  static std::vector<IdPairs> getExpectedPCLSuccess() {
    return {
      {{ 3, 13 }, { 4, 14 }, { 2, 12 }, { 5, 15 }}
    };
  }
};

#endif // SEGMATCH_RECOGNIZER_TEST_DATA_HPP_
