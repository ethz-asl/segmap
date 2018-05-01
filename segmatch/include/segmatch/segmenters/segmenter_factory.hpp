#ifndef SEGMATCH_SEGMENTER_FACTORY_HPP_
#define SEGMATCH_SEGMENTER_FACTORY_HPP_

#include "segmatch/segmatch.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

/// \brief Factory class for segmenters.
class SegmenterFactory {
 public:
  /// \brief Initializes a new instance of the SegmenterFactory class.
  /// \param params The current parameters of SegMatch.
  SegmenterFactory(SegMatchParams params);

  /// \brief Creates a segmenter.
  /// \returns Pointer to a new CorrespondencdRecognizer instance.
  std::unique_ptr<Segmenter<MapPoint>> create() const;

 private:
  SegmenterParameters params_;
}; // class SegmenterFactory

} // namespace segmatch

#endif // SEGMATCH_SEGMENTER_FACTORY_HPP_
