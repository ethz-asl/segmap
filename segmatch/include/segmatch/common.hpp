#ifndef SEGMATCH_COMMON_HPP_
#define SEGMATCH_COMMON_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "segmatch/point_extended.hpp"

namespace segmatch {

typedef segmatch::PointExtended MapPoint;
typedef pcl::PointCloud<MapPoint> MapCloud;
typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointICloud;
typedef std::pair<PointICloud, PointICloud> PointICloudPair;
typedef PointICloud::Ptr PointICloudPtr;
typedef std::pair<PointI, PointI> PointIPair;
typedef std::vector<PointIPair> PointIPairs;

typedef pcl::PointXYZ PclPoint;
typedef pcl::PointCloud<PclPoint> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef std::pair<PclPoint, PclPoint> PointPair;
typedef std::vector<PointPair> PointPairs;

typedef pcl::Normal PclNormal;
typedef pcl::PointCloud<PclNormal> PointNormals;
typedef pcl::PointCloud<PclNormal>::Ptr PointNormalsPtr;

typedef pcl::PointCloud<pcl::FPFHSignature33> PointDescriptors;
typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr PointDescriptorsPtr;

typedef std::vector<Eigen::Matrix4f,
    Eigen::aligned_allocator<Eigen::Matrix4f> > RotationsTranslations;

typedef std::pair<std::string, std::string> FilenamePair;

typedef std::pair<PointICloudPtr, PointICloudPtr> CloudPair;

/*
 * \brief Type representing IDs, for example for segments or clouds
 * Warning: the current implementation sometimes uses IDs as array indices.
 */
typedef int64_t Id;
const Id kNoId = -1;
const Id kInvId = -2;
const Id kUnassignedId = -3; // Used internally by the incremental segmenter.

// TODO(Renaud @ Daniel) this is probably not the best name but we need to have this format
// Somewhere as the classifier output matches between two samples and the geometric also
// takes pairs. It collides a bit with IdMatches. We can discuss that. I also added for
// convenience the centroids in there as they are easy to get when grabbing the Ids.
// Let's see how that evolves.
// 
// TODO: switch to std::array of size 2? so that the notation is the same .at(0) instead of first.
typedef std::pair<Id, Id> IdPair;

class PairwiseMatch {
 public:
  PairwiseMatch(Id id1, Id id2, const PclPoint& centroid1, const PclPoint& centroid2,
                float confidence_in) :
                  ids_(id1, id2),
                  confidence_(confidence_in),
                  centroids_(PointPair(centroid1, centroid2)) {}

  PointPair getCentroids() const { return centroids_; }
  IdPair ids_;
  float confidence_;
  Eigen::MatrixXd features1_;
  Eigen::MatrixXd features2_;
  PointPair centroids_;
};

typedef std::vector<PairwiseMatch,
    Eigen::aligned_allocator<PairwiseMatch> > PairwiseMatches;
    
struct Translation {
  Translation(double x_in, double y_in, double z_in) :
    x(x_in), y(y_in), z(z_in) {}
  double x;
  double y;
  double z;
};

} // namespace segmatch

#endif // SEGMATCH_COMMON_HPP_
