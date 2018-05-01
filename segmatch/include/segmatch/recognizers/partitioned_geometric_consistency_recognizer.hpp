#ifndef SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
#define SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_

#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/graph_based_geometric_consistency_recognizer.hpp"

namespace segmatch {

/// \brief Recognizes a model in a scene using a graph-based approach. First a consistency graph
/// is constructed, where the nodes represent the matches and edges connect matches that are
/// pairwise consistent. Recognition finds a maximum clique matches that are pairwise consistent.
/// The partitioned approach assumes that the model is relatively small compared to the scene. This
/// allows to partition the space in a voxel grid with resolution equal to the size of the model
/// and efficiently discard all pairwise matches that are two or more voxels apart, i.e. too
/// distant to be consistent given the model size.
/// \remark The current implementation assumes that the model and the scene are almost planar,
/// and partitioning is necessary only on the X and Y axis. Extension to 3D is possible by using
/// 3D partitioning.
class PartitionedGeometricConsistencyRecognizer : public GraphBasedGeometricConsistencyRecognizer {
 public:
  /// \brief Initializes a new instance of the PartitionedGeometricConsistencyRecognizer class.
  /// \param params The parameters of the geometry consistency grouping.
  /// \param max_model_radius Radius of the bounding cylinder of the model.
  PartitionedGeometricConsistencyRecognizer(const GeometricConsistencyParams& params,
                                            float max_model_radius) noexcept;

 protected:
  /// \brief Builds a consistency graph of the provided matches.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  /// \returns Graph encoding pairwise consistencies. Match \c predicted_matches[i] is represented
  /// by node \c i .
  ConsistencyGraph buildConsistencyGraph(const PairwiseMatches& predicted_matches) override;

 private:
  // Per-partition data
  struct PartitionData { };

  // Find consistencies within a partition and add them to the consistency graph.
  size_t findAndAddInPartitionConsistencies(const PairwiseMatches& predicted_matches,
                                          const std::vector<size_t>& partition_indices,
                                          ConsistencyGraph& consistency_graph) const;

  // Find consistencies between two partitions and add them to the consistency graph.
  size_t findAndAddCrossPartitionConsistencies(const PairwiseMatches& predicted_matches,
                                             const std::vector<size_t>& partition_indices_1,
                                             const std::vector<size_t>& partition_indices_2,
                                             ConsistencyGraph& consistency_graph) const;

  float partition_size_;
}; // class PartitionedGeometricConsistencyRecognizer

} // namespace segmatch

#endif // SEGMATCH_PARTITIONED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
