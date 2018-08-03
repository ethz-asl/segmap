#include "segmatch/recognizers/graph_based_geometric_consistency_recognizer.hpp"

#include <algorithm>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>

#include "segmatch/common.hpp"
#include "segmatch/recognizers/graph_utilities.hpp"

namespace segmatch {

GraphBasedGeometricConsistencyRecognizer::GraphBasedGeometricConsistencyRecognizer(
    const GeometricConsistencyParams& params) noexcept
  : params_(params) {
}

void GraphBasedGeometricConsistencyRecognizer::recognize(
    const PairwiseMatches& predicted_matches) {
  // Clear the current candidates and check if we got matches.
  candidate_transfomations_.clear();
  candidate_matches_.clear();
  if (predicted_matches.empty()) return;

  // Build a graph encoding consistencies between the predicted matches.
  ConsistencyGraph consistency_graph = buildConsistencyGraph(predicted_matches);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.NumConsistencies",
                         boost::num_edges(consistency_graph));

  BENCHMARK_START("SM.Worker.Recognition.FindClique");
  std::vector<size_t> maximum_clique =  GraphUtilities::findMaximumClique(
      consistency_graph, params_.min_cluster_size);
  BENCHMARK_STOP("SM.Worker.Recognition.FindClique");

  if (maximum_clique.empty()) return;

  // Store the maximum clique of matches found.
  candidate_matches_.emplace_back();
  candidate_matches_.back().reserve(maximum_clique.size());
  for (const auto match_index : maximum_clique) {
    candidate_matches_.back().push_back(predicted_matches[match_index]);
  }

  // Estimate the 3D transformation between model and scene.
  Eigen::Matrix4f transformation = estimateRigidTransformation(candidate_matches_.back());
  candidate_transfomations_.push_back(transformation);
}

inline Eigen::Matrix4f GraphBasedGeometricConsistencyRecognizer::estimateRigidTransformation(
    const PairwiseMatches& true_matches) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.ComputeTransformation");

  // We limit the number of matches for estimating the transform to 8 as pcl::umeyama sometimes
  // crashes with 10+ matches.
  const unsigned int n_matches_to_consider = std::min(
      static_cast<unsigned>(true_matches.size()), 8u);
  
  Eigen::Matrix<double, 3, Eigen::Dynamic> source(3, n_matches_to_consider);
  Eigen::Matrix<double, 3, Eigen::Dynamic> target(3, n_matches_to_consider);

  for (size_t i = 0u; i < n_matches_to_consider; ++i) {
    source(0, i) = static_cast<float>(true_matches[i].centroids_.first.x);
    source(1, i) = static_cast<float>(true_matches[i].centroids_.first.y);
    source(2, i) = static_cast<float>(true_matches[i].centroids_.first.z);

    target(0, i) = static_cast<float>(true_matches[i].centroids_.second.x);
    target(1, i) = static_cast<float>(true_matches[i].centroids_.second.y);
    target(2, i) = static_cast<float>(true_matches[i].centroids_.second.z);
  }

  // Estimate rigid transform using the least squared umeyama method. "Least-squares estimation of
  // transformation parameters between two point patterns", Shinji Umeyama, DOI: 10.1109/34.88573
  return pcl::umeyama(source, target, false).cast<float>();
}

} // namespace segmatch
