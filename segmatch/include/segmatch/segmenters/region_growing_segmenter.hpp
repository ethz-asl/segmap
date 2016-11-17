#ifndef SEGMATCH_REGION_GROWING_SEGMENTER_HPP_
#define SEGMATCH_REGION_GROWING_SEGMENTER_HPP_

#include <string>

#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/region_growing.h>

#include "segmatch/parameters.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

class RegionGrowingSegmenter : public Segmenter {
 public:
  RegionGrowingSegmenter();
  explicit RegionGrowingSegmenter(const SegmenterParameters& params);
  ~RegionGrowingSegmenter();

  /// \brief Segment the point cloud.
  virtual void segment(const PointICloud& cloud,
                       SegmentedCloud* segmented_cloud);

 private:
  SegmenterParameters params_;

  // PCL object members.
  pcl::search::KdTree<PointI>::Ptr kd_tree_;
  pcl::NormalEstimationOMP<PointI, pcl::Normal> normal_estimator_omp_;
  pcl::RegionGrowing<PointI, pcl::Normal> region_growing_estimator_;
}; // class RegionGrowingSegmenter

} // namespace segmatch

#endif // SEGMATCH_REGION_GROWING_SEGMENTER_HPP_
