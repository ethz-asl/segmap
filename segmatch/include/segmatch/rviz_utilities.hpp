#ifndef SEGMATCH_RVIZ_UTILITIES_HPP_
#define SEGMATCH_RVIZ_UTILITIES_HPP_

#include "segmented_cloud.hpp"

namespace segmatch {

class RVizUtilities {
 public:
  /// \brief \c RVizUtilities is a static class only. Forbid instantiation.
  RVizUtilities() = delete;

  /// \brief Converts a segmented cloud to a \c PointICloud that can be used
  /// for visualization purposes in RViz.
  /// \remark It is recommended to disable "Autocompute Intensity Bounds" and
  /// set "Max Intensity" to 4096 in RViz.
  static PointICloud segmentedCloudtoPointICloud(
      const SegmentedCloud& segmented_cloud, bool use_point_cloud_to_publish = false,
      bool use_reconstruction = false);

  /// \brief Converts a segmented cloud to a \c PointICloud where semgents are colored by
  /// semantic information.
  static PointICloud segmentedCloudSemanticstoPointICloud(
      const SegmentedCloud& segmented_cloud, bool use_reconstruction = false,
      bool get_compressed = false);

 private:
  // Computes an intensity value that represents the segment color. Intensities
  // lie in the range [0, 4096] and depend on the segment ID. It is recommended
  // to disable "Autocompute Intensity Bounds" and set "Max Intensity" to 4096
  // in RViz.
  static float getSegmentColorAsIntensity(const Id segment_id);
}; // class RVizUtilities

} // namespace segmatch

#endif // SEGMATCH_RVIZ_UTILITIES_HPP_
