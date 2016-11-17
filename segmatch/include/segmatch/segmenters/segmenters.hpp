#ifndef SEGMATCH_SEGMENTERS_HPP_
#define SEGMATCH_SEGMENTERS_HPP_

#include "segmatch/segmenters/euclidean_segmenter.hpp"
#include "segmatch/segmenters/region_growing_segmenter.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

static std::unique_ptr<Segmenter> create_segmenter(const SegmenterParameters& parameters) {
  std::unique_ptr<Segmenter> segmenter;
  if (parameters.segmenter_type == "RegionGrowingSegmenter") {
    segmenter = std::unique_ptr<Segmenter>(new RegionGrowingSegmenter(parameters));
  } else if (parameters.segmenter_type == "EuclideanSegmenter") {
    segmenter = std::unique_ptr<Segmenter>(new EuclideanSegmenter(parameters));
  } else {
    LOG(FATAL) << "The segmenter " << parameters.segmenter_type << " was not implemented.";
  }
  return segmenter;
}

} // namespace segmatch

#endif // SEGMATCH_SEGMENTERS_HPP_
