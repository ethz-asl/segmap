#include "segmatch/recognizers/partitioned_geometric_consistency_recognizer.hpp"

#include <algorithm>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>

#include "segmatch/common.hpp"
#include "segmatch/recognizers/graph_utilities.hpp"
#include "segmatch/recognizers/matches_partitioner.hpp"

namespace segmatch {

PartitionedGeometricConsistencyRecognizer::PartitionedGeometricConsistencyRecognizer(
    const GeometricConsistencyParams& params, float max_model_radius) noexcept
  : GraphBasedGeometricConsistencyRecognizer(params), partition_size_(max_model_radius * 2.0f) {
}

inline size_t PartitionedGeometricConsistencyRecognizer::findAndAddInPartitionConsistencies(
    const PairwiseMatches& predicted_matches, const std::vector<size_t>& partition_indices,
    ConsistencyGraph& consistency_graph) const {
  size_t num_consistency_tests = 0u;
  for (size_t i2 = 0u; i2 < partition_indices.size(); ++i2) {
    size_t i = partition_indices[i2];
    const Eigen::Vector3f& scene_point_i = predicted_matches[i].centroids_.first.getVector3fMap();
    const Eigen::Vector3f& model_point_i = predicted_matches[i].centroids_.second.getVector3fMap();
    for (size_t j2 = i2 + 1u; j2 < partition_indices.size(); ++j2) {
      size_t j = partition_indices[j2];
      const Eigen::Vector3f& scene_point_j =
          predicted_matches[j].centroids_.first.getVector3fMap();
      const Eigen::Vector3f& model_point_j =
          predicted_matches[j].centroids_.second.getVector3fMap();

      // Compute the difference between the distances and add edge to the consistency graph if the
      // matches are consistent.
      ++num_consistency_tests;
      float dist_scene = (scene_point_i - scene_point_j).norm();
      float dist_model = (model_point_i - model_point_j).norm();
      float distance = fabs(dist_scene - dist_model);
      if (distance <= static_cast<float>(params_.resolution))
        boost::add_edge(i, j, consistency_graph);
    }
  }
  return num_consistency_tests;
}

inline size_t PartitionedGeometricConsistencyRecognizer::findAndAddCrossPartitionConsistencies(
    const PairwiseMatches& predicted_matches, const std::vector<size_t>& partition_indices_1,
    const std::vector<size_t>& partition_indices_2, ConsistencyGraph& consistency_graph) const {
  size_t num_consistency_tests = 0u;
  for (size_t i2 = 0u; i2 < partition_indices_1.size(); ++i2) {
    size_t i = partition_indices_1[i2];
    const Eigen::Vector3f& scene_point_i = predicted_matches[i].centroids_.first.getVector3fMap();
    const Eigen::Vector3f& model_point_i = predicted_matches[i].centroids_.second.getVector3fMap();
    for (size_t j2 = 0u; j2 < partition_indices_2.size(); ++j2) {
      size_t j = partition_indices_2[j2];
      const Eigen::Vector3f& scene_point_j =
          predicted_matches[j].centroids_.first.getVector3fMap();
      const Eigen::Vector3f& model_point_j =
          predicted_matches[j].centroids_.second.getVector3fMap();

      // Compute the difference between the distances and add edge to the consistency graph if the
      // matches are consistent.
      ++num_consistency_tests;
      float dist_scene = (scene_point_i - scene_point_j).norm();
      float dist_model = (model_point_i - model_point_j).norm();
      float distance = fabs(dist_scene - dist_model);
      if (distance <= static_cast<float>(params_.resolution))
        boost::add_edge(i, j, consistency_graph);
    }
  }
  return num_consistency_tests;
}

PartitionedGeometricConsistencyRecognizer::ConsistencyGraph
PartitionedGeometricConsistencyRecognizer::buildConsistencyGraph(
    const PairwiseMatches& predicted_matches) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph");

  // Partition the matches in a grid by the position of the scene points. The size of the
  // partitions is greater or equal the size of the model. This way we can safely assume that, if
  // the model is actually present in the scene, all matches will be contained in a 2x2 group of
  // adjacent partitions.
  BENCHMARK_START("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");
  MatchesGridPartitioning<PartitionData> partitioning =
      MatchesPartitioner::computeGridPartitioning<PartitionData>(predicted_matches,
                                                                 partition_size_);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.NumPartitions",
                         partitioning.getHeight() * partitioning.getWidth());
  BENCHMARK_STOP("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");
  ConsistencyGraph consistency_graph(predicted_matches.size());
  size_t num_consistency_tests = 0u;

  // Find all possible consistency within a partition and within neighbor partitions.
  for (size_t i = 0; i < partitioning.getHeight(); ++i) {
    for (size_t j = 0; j < partitioning.getWidth(); ++j) {
      // Find in-partition consistencies and add them to the consistency graph.
      findAndAddInPartitionConsistencies(
          predicted_matches, partitioning(i, j).match_indices, consistency_graph);

      // Determine which neighbor partitions exist.
      bool has_right_neighbors = j < partitioning.getWidth() - 1u;
      bool has_bottom_neighbors = i < partitioning.getHeight() - 1u;
      bool has_left_neighbors = j > 0u;

      // Find possible cross-partition consistencies and add them to the consistency graph.
      if (has_right_neighbors) {
        num_consistency_tests += findAndAddCrossPartitionConsistencies(
            predicted_matches, partitioning(i, j).match_indices,
            partitioning(i, j + 1u).match_indices, consistency_graph);
      }
      if (has_right_neighbors && has_bottom_neighbors) {
        num_consistency_tests += findAndAddCrossPartitionConsistencies(
            predicted_matches, partitioning(i, j).match_indices,
            partitioning(i + 1u, j + 1u).match_indices, consistency_graph);
      }
      if (has_bottom_neighbors) {
        num_consistency_tests += findAndAddCrossPartitionConsistencies(
            predicted_matches, partitioning(i, j).match_indices,
            partitioning(i + 1u, j).match_indices, consistency_graph);
      }
      if (has_bottom_neighbors && has_left_neighbors) {
        num_consistency_tests += findAndAddCrossPartitionConsistencies(
            predicted_matches, partitioning(i, j).match_indices,
            partitioning(i + 1u, j - 1u).match_indices, consistency_graph);
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.NumConsistencyTests",
                         num_consistency_tests);

  return consistency_graph;
}

} // namespace segmatch
