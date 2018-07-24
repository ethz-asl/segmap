#include <segmatch/segmenters/impl/smoothness_constraints_segmenter.hpp>
#include "segmatch/common.hpp"

namespace segmatch {
// Instantiate SmoothnessConstraintsSegmenter for the template parameters used in the application.
template class SmoothnessConstraintsSegmenter<MapPoint>;
// Add any other required instantiation here or in a separate file and declare them in
// segmatch/segmenters/impl/smoothness_constraints_segmenter.hpp.
} // namespace segmatch
