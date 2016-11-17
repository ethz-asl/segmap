#ifndef SEGMATCH_SEGMENTER_HPP_
#define SEGMATCH_SEGMENTER_HPP_

#include <string>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class Segmenter {
 public:
  /// \brief Segment the point cloud.
  virtual void segment(const PointICloud& cloud,
                       SegmentedCloud* segmented_cloud) = 0;

}; // class Segmenter

} // namespace segmatch

#endif // SEGMATCH_SEGMENTER_HPP_
