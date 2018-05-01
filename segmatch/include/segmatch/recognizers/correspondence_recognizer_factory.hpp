#ifndef SEGMATCH_CORRESPONDENCE_RECOGNIZER_FACTORY_HPP_
#define SEGMATCH_CORRESPONDENCE_RECOGNIZER_FACTORY_HPP_

#include "segmatch/segmatch.hpp"
#include "segmatch/recognizers/correspondence_recognizer.hpp"

namespace segmatch {

/// \brief Factory class for correspondence recognizers.
class CorrespondenceRecognizerFactory {
 public:
  /// \brief Initializes a new instance of the CorrespondenceRecognizerFactory class.
  /// \param params The current parameters of SegMatch.
  CorrespondenceRecognizerFactory(const SegMatchParams& params);

  /// \brief Creates a correspondence recognizer.
  /// \returns Pointer to a new CorrespondencdRecognizer instance.
  std::unique_ptr<CorrespondenceRecognizer> create() const;

 private:
  GeometricConsistencyParams params_;
  float local_map_radius_;
}; // class CorrespondenceRecognizerFactory

} // namespace segmatch

#endif // SEGMATCH_CORRESPONDENCE_RECOGNIZER_FACTORY_HPP_
