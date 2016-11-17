#ifndef SEGMATCH_EUCLIDEAN_SEGMENTER_HPP_
#define SEGMATCH_EUCLIDEAN_SEGMENTER_HPP_

#include <string>

#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "segmatch/parameters.hpp"
#include "segmatch/segmenters/segmenter.hpp"

namespace segmatch {

class EuclideanSegmenter : public Segmenter {
 public:
  EuclideanSegmenter();
  explicit EuclideanSegmenter(const SegmenterParameters& params);
  ~EuclideanSegmenter();

  /// \brief Segment the point cloud.
  virtual void segment(const PointICloud& cloud,
                       SegmentedCloud* segmented_cloud);

 private:
  SegmenterParameters params_;

  // PCL object members.
  pcl::search::KdTree<PointI>::Ptr kd_tree_;
  pcl::EuclideanClusterExtraction<PointI> euclidean_cluster_extractor_;
}; // class EuclideanSegmenter

} // namespace segmatch

#endif // SEGMATCH_EUCLIDEAN_SEGMENTER_HPP_
