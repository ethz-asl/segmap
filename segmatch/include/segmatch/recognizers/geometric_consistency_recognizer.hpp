#ifndef SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
#define SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_

#include "segmatch/parameters.hpp"
#include "segmatch/recognizers/correspondence_recognizer.hpp"

namespace segmatch {

/// \brief Recognizes a model in a scene using the \c GeometricConsistencyRecognizer implementation
/// provided by the PCL. Recognition finds a subset of centroids in the model whose pairwise
/// distances are consistent with the matching centroids in the scene.
class GeometricConsistencyRecognizer : public CorrespondenceRecognizer {
 public:
  /// \brief Initializes a new instance of the GeometricConsistencyRecognizer class.
  /// \param params The parameters of the geometry consistency grouping.
  GeometricConsistencyRecognizer(const GeometricConsistencyParams& params) noexcept
    : params_(params) { }

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

 private:
  // Candidate transformations and matches between model and scene.
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  candidate_transfomations_;
  std::vector<PairwiseMatches> candidate_matches_;

  // The parameters of the geometry consistency grouping.
  GeometricConsistencyParams params_;
}; // class GeometricConsistencyRecognizer

} // namespace segmatch

#endif // SEGMATCH_GEOMETRIC_CONSISTENCY_RECOGNIZER_HPP_
