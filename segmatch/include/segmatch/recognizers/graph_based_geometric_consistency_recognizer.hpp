#ifndef SEGMATCH_GRAPH_BASED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
#define SEGMATCH_GRAPH_BASED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_

#include <boost/graph/adjacency_list.hpp>

#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/correspondence_recognizer.hpp"

namespace segmatch {

/// \brief Recognizes a model in a scene using a graph-based approach. First a consistency graph
/// is constructed, where the nodes represent the matches and edges connect matches that are
/// pairwise consistent. Recognition finds a maximum clique matches that are pairwise consistent.
class GraphBasedGeometricConsistencyRecognizer : public CorrespondenceRecognizer {
 public:
  /// \brief Initializes a new instance of the GraphBasedGeometricConsistencyRecognizer class.
  /// \param params The parameters of the geometry consistency grouping.
  explicit GraphBasedGeometricConsistencyRecognizer(
      const GeometricConsistencyParams& params) noexcept;

  /// \brief Sets the current matches and tries to recognize the model.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  void recognize(const PairwiseMatches& predicted_matches) override;

  /// \brief Gets the candidate transformations between model and scene.
  /// \returns Vector containing the candidate transformations. Transformations are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  getCandidateTransformations() const override {
    return candidate_transfomations_;
  }

  /// \brief Gets the candidate clusters of matches between model and scene. Every cluster
  /// represents a possible recognition.
  /// \returns Vector containing the candidate clusters. Clusters are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  const std::vector<PairwiseMatches>& getCandidateClusters() const override {
    return candidate_matches_;
  }

 protected:
  // Data types for the consistency graph.
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> ConsistencyGraph;

  /// \brief Builds a consistency graph of the provided matches.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  /// \returns Graph encoding pairwise consistencies. Match \c predicted_matches[i] is represented
  /// by node \c i .
  virtual ConsistencyGraph buildConsistencyGraph(const PairwiseMatches& predicted_matches) = 0;

  // The parameters of the geometry consistency grouping.
  GeometricConsistencyParams params_;

 private:
  // Estimate 3D transform between model and scene.
  Eigen::Matrix4f estimateRigidTransformation(const PairwiseMatches& true_matches);

  // Candidate transformations and matches between model and scene.
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  candidate_transfomations_;
  std::vector<PairwiseMatches> candidate_matches_;
}; // class GraphBasedGeometricConsistencyRecognizer

} // namespace segmatch

#endif // SEGMATCH_GRAPH_BASED_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
