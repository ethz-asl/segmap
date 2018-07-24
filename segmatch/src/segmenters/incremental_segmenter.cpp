#include <segmatch/segmenters/region_growing_policy.hpp>
#include "segmatch/segmenters/impl/incremental_segmenter.hpp"

#include "segmatch/common.hpp"

namespace segmatch {
// Instantiate IncrementalEuclideanSegmenter for the template parameters used in the application.
template class IncrementalSegmenter<MapPoint, EuclideanDistance>;
template class IncrementalSegmenter<MapPoint, SmoothnessConstraints>;
// Add any other required instantiation here or in a separate file and declare them in
// segmatch/segmenters/impl/incremental_segmenter.hpp.
} // namespace segmatch
