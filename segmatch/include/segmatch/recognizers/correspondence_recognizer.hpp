#ifndef SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_
#define SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_

#include <vector>

#include "segmatch/common.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

/// \brief Base class for recognizing a model in a scene.
class CorrespondenceRecognizer {
 public:
  /// \brief Finalizes an instance of the CorrespondenceRecognizer class.
  virtual ~CorrespondenceRecognizer() = default;

  /// \brief Sets the current matches and tries to recognize the model.
  /// \param predicted_matches Vector of possible correspondences between model and scene.
  virtual void recognize(const PairwiseMatches& predicted_matches) = 0;

  /// \brief Gets the candidate transformations between model and scene.
  /// \returns Vector containing the candidate transformations. Transformations are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  virtual const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  getCandidateTransformations() const = 0;

  /// \brief Gets the candidate clusters of matches between model and scene. Every cluster
  /// represents a possible recognition.
  /// \returns Vector containing the candidate clusters. Clusters are sorted in
  /// decreasing recognition quality order. If empty, the model was not recognized.
  virtual const std::vector<PairwiseMatches>& getCandidateClusters() const = 0;
}; // class CorrespondenceRecognizer

} // namespace segmatch

#endif // SEGMATCH_CORRESPONDENCE_RECOGNIZER_HPP_
