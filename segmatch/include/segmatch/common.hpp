#ifndef SEGMATCH_COMMON_HPP_
#define SEGMATCH_COMMON_HPP_

#include <limits>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <laser_slam/common.hpp>
#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

namespace segmatch {

typedef kindr::minimal::QuatTransformationTemplate<double> SE3;

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

struct FeaturedCloud {
  //  std::string category;
  //  double resolution;
  FeaturedCloud() : keypoints(new PointCloud()),
      descriptors(new PointDescriptors()) {}
  ~FeaturedCloud() {
    keypoints.reset();
    descriptors.reset();
  }

  PointCloudPtr keypoints;
  PointDescriptorsPtr descriptors;
};

typedef std::vector<Eigen::Matrix4f,
    Eigen::aligned_allocator<Eigen::Matrix4f> > RotationsTranslations;
typedef std::vector<pcl::Correspondences> Correspondences;

typedef std::pair<std::string, std::string> FilenamePair;

typedef std::pair<PointICloudPtr, PointICloudPtr> CloudPair;

// Load cloud pair from disk.
static CloudPair loadCloudPair(const FilenamePair& filename_pair) {
  CloudPair cloud_pair(PointICloudPtr(new PointICloud),
                       PointICloudPtr(new PointICloud));
  LOG(INFO) << "Loading first point cloud: " << filename_pair.first;
  CHECK_NE(pcl::io::loadPCDFile(filename_pair.first, *(cloud_pair.first)), -1) <<
      "Failed to load first point cloud.";
  LOG(INFO) << "Loading second point cloud: " << filename_pair.second;
  CHECK_NE(pcl::io::loadPCDFile(filename_pair.second, *(cloud_pair.second)), -1) <<
      "Failed to load second point cloud.";
  return cloud_pair;
}

/*
 * \brief Type representing IDs, for example for segments or clouds
 * Warning: the current implementation sometimes uses IDs as array indices.
 */
typedef int64_t Id;
const Id kNoId = -1;
const Id kInvId = -2;

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
typedef std::vector<PairwiseMatch> PairwiseMatches;

/*
 * Struct associating a segment id with a counter for this id
 * used in calculating overlapping segments.
 */
struct IdCounter {
  Id id = kNoId;
  // Number of points associated to segment at id.
  unsigned int count = 0u;
  // Total number of points which were tested for association.
  unsigned int total_count = 0u;
};

/*
 * An Id paired with an index.
 */
struct IdIndex {
  Id id = kNoId;
  size_t index = 0u;
};

static void loadCloud(const std::string& filename, PointICloud* cloud) {
  LOG(INFO) <<"Loading cloud: " << filename << ".";
  CHECK_NE(pcl::io::loadPCDFile(filename, *cloud), -1) <<
      "Failed to load cloud: " << filename << ".";
}

struct Translation {
  Translation(double x_in, double y_in, double z_in) :
    x(x_in), y(y_in), z(z_in) {}
  double x;
  double y;
  double z;
};

static void translateCloud(const Translation& translation, PointICloud* cloud) {
  for (size_t i = 0u; i < cloud->size(); ++i) {
    cloud->points[i].x += translation.x;
    cloud->points[i].y += translation.y;
    cloud->points[i].z += translation.z;
  }
}

static PointPair findLimitPoints(const PointICloud& cloud) {
  PclPoint min_point, max_point;
  min_point.x = std::numeric_limits<float>::max();
  min_point.y = std::numeric_limits<float>::max();
  min_point.z = std::numeric_limits<float>::max();

  max_point.x = std::numeric_limits<float>::min();
  max_point.y = std::numeric_limits<float>::min();
  max_point.z = std::numeric_limits<float>::min();

  for (size_t i = 0u; i < cloud.size(); ++i) {
    if (cloud.points[i].x < min_point.x) {
      min_point.x = cloud.points[i].x;
    }
    if (cloud.points[i].y < min_point.y) {
      min_point.y = cloud.points[i].y;
    }
    if (cloud.points[i].z < min_point.z) {
      min_point.z = cloud.points[i].z;
    }
    if (cloud.points[i].x > max_point.x) {
      max_point.x = cloud.points[i].x;
    }
    if (cloud.points[i].y > max_point.y) {
      max_point.y = cloud.points[i].y;
    }
    if (cloud.points[i].z > max_point.z) {
      max_point.z = cloud.points[i].z;
    }
  }
  return PointPair(min_point, max_point);
}

static void extractBox(const PointPair& limit_points, float margin, PointICloud* cloud) {
  CHECK_NOTNULL(cloud);
  PointICloud filtered_cloud;
  for (size_t i = 0u; i < cloud->size(); ++i) {
    if (cloud->points[i].x > limit_points.first.x - margin &&
        cloud->points[i].x < limit_points.second.x + margin &&
        cloud->points[i].y > limit_points.first.y - margin &&
        cloud->points[i].y < limit_points.second.y + margin &&
        cloud->points[i].z > limit_points.first.z - margin &&
        cloud->points[i].z < limit_points.second.z + margin) {
      filtered_cloud.points.push_back(cloud->points[i]);
    }
  }
  LOG(INFO) << "Extracting box from " << cloud->size() << " points to " << filtered_cloud.size() << " points.";
  *cloud = filtered_cloud;
}

static void applyCylindricalFilter(const PclPoint& center, double radius_m,
                                   double height_m, PointICloud* cloud) {
  CHECK_NOTNULL(cloud);
  PointICloud filtered_cloud;

  const double radius_squared = pow(radius_m, 2.0);
  const double height_halved_m = height_m / 2.0;

  for (size_t i = 0u; i < cloud->size(); ++i) {
    if ((pow(cloud->points[i].x - center.x, 2.0)
        + pow(cloud->points[i].y - center.y, 2.0)) <= radius_squared &&
        abs(cloud->points[i].z - center.z) <= height_halved_m) {
      filtered_cloud.points.push_back(cloud->points[i]);
    }
  }

  filtered_cloud.width = 1;
  filtered_cloud.height = filtered_cloud.points.size();

  ROS_INFO_STREAM("Applied cylindrical filter from " << cloud->size()
                  << " points to " << filtered_cloud.size() << " points.");
  *cloud = filtered_cloud;
}

// Find the (index of the) nearest neighbour of point in target_cloud.
static bool findNearestNeighbour(const PclPoint& point, const PointCloud& target_cloud,
                                 size_t* index, float* squared_distance = NULL) {
  CHECK_NOTNULL(index);

  // Set up nearest neighbour search.
  pcl::KdTreeFLANN<PclPoint> kdtree;
  PointCloudPtr target_cloud_copy_ptr(new PointCloud);
  pcl::copyPointCloud(target_cloud, *target_cloud_copy_ptr);
  kdtree.setInputCloud(target_cloud_copy_ptr);
  std::vector<int> nearest_neighbour_indices(1);
  std::vector<float> nearest_neighbour_squared_distances(1);

  // Find the nearest neighbours in target.
  if (kdtree.nearestKSearch(point, 1, nearest_neighbour_indices,
                            nearest_neighbour_squared_distances) <= 0) {
    LOG(ERROR) << "Nearest neighbour search failed.";
    return false;
  }

  // Return values.
  *index = nearest_neighbour_indices.at(0);
  if (squared_distance != NULL) { *squared_distance = nearest_neighbour_squared_distances.at(0); }
  return true;
}

static bool findNearestNeighbour(const PointI& point, const PointICloud& target_cloud,
                                 size_t* index, float* squared_distance = NULL) {
  // Convert inputs.
  // TODO: templates would be better.
  PclPoint point_converted;
  point_converted.x = point.x;
  point_converted.y = point.y;
  point_converted.z = point.z;
  PointCloud target_cloud_converted;
  pcl::copyPointCloud(target_cloud, target_cloud_converted);
  // Use pre-existing function.
  return findNearestNeighbour(point_converted, target_cloud_converted, index, squared_distance);
}

static PairwiseMatches filterNonReciprocalMatches(const PairwiseMatches& first,
                                                  const PairwiseMatches& second) {
  PairwiseMatches result;
  for (size_t i = 0u; i < first.size(); ++i) {
    // Look for reciprocal match in second.
    for (size_t j = 0u; j < second.size(); ++j) {
      if (first.at(i).ids_.second == second.at(j).ids_.first &&
          first.at(i).ids_.first == second.at(j).ids_.second) {
        result.push_back(first.at(i));
      }
    }
  }
  return result;
}

static void displayPerformances(unsigned int tp, unsigned int tn,
                                unsigned int fp, unsigned int fn) {

  LOG(INFO) << "TP: " << tp << ", TN: " << tn <<
      ", FP: " << fp << ", FN: " << fn << ".";

  const double true_positive_rate = double(tp) / double(tp + fn);
  const double true_negative_rate = double(tn) / double(fp + tn);
  const double false_positive_rate = 1.0 - true_negative_rate;

  LOG(INFO) << "Accuracy (ACC): " << double(tp + tn) /
      double(tp + fp + tn + fn);
  LOG(INFO) << "Sensitivity (TPR): " << true_positive_rate;
  LOG(INFO) << "Specificity (TNR): " << true_negative_rate;
  LOG(INFO) << "Precision: " << double(tp) / double(tp + fp);
  LOG(INFO) << "Positive likelyhood ratio: " << true_positive_rate / false_positive_rate;
}

template <typename T>
inline bool in(const T& obj, const std::vector<T>& vec, size_t* index = NULL) {
  for (size_t i = 0u; i < vec.size(); ++i) {
    if (obj == vec.at(i)) {
      if (index != NULL) { *index = i; }
      return true;
    }
  }
  return false;
}

template <typename T>
inline std::vector<T>& removeDuplicates(const std::vector<T>& vec, size_t* n_removed = NULL) {
  if (n_removed != NULL) { *n_removed = 0u; }
  std::vector<T> result;
  for (size_t i = 0u; i < vec.size(); ++i) {
    if (!in(vec.at(i), result)) {
      result.push_back(vec.at(i));
    } else if (n_removed != NULL) {
      (*n_removed)++;
    }
  }
  return result;
}

static bool isValidLoopClosure(const Eigen::Matrix4f& recommended_transformation,
                               const double threshold_on_disparity_angle_for_true_positive,
                               const double threshold_on_distance_for_true_positive) {
  std::cout << "Transformation matrix " << std::endl << recommended_transformation << std::endl;
  Eigen::Matrix<double, 3, 3> rotation_matrix = recommended_transformation.block<3,3>(0,0).cast<double>();
  laser_slam::SO3 rotation = laser_slam::SO3::fromApproximateRotationMatrix(rotation_matrix);
  double disparity_angle = rotation.getDisparityAngle(laser_slam::SO3());
  std::cout << "disparity_angle: " << disparity_angle << std::endl;
  //TODO better this and add check on translation as well.
  if (disparity_angle >= threshold_on_disparity_angle_for_true_positive) {
    return false;
  } else {
    if ((recommended_transformation(0,3) * recommended_transformation(0,3) +
        recommended_transformation(1,3) * recommended_transformation(1,3))
        > threshold_on_distance_for_true_positive * threshold_on_distance_for_true_positive) {
      return false;
    } else {
      return true;
    }
  }
}

static PointCloud lpmToPcl(const laser_slam::PointMatcher::DataPoints& cloud_in) {
  PointCloud cloud_out;
  cloud_out.width = cloud_in.getNbPoints();
  cloud_out.height = 1;
  for (size_t i = 0u; i < cloud_in.getNbPoints(); ++i) {
    PclPoint point;
    point.x = cloud_in.features(0,i);
    point.y = cloud_in.features(1,i);
    point.z = cloud_in.features(2,i);
    cloud_out.push_back(point);
  }
  return cloud_out;
}

static void applyCylindricalFilter(const PclPoint& center, double radius_m,
                                   double height_m, bool remove_point_inside,
                                   PointCloud* cloud) {
  CHECK_NOTNULL(cloud);
  PointCloud filtered_cloud;

  const double radius_squared = pow(radius_m, 2.0);
  const double height_halved_m = height_m / 2.0;

  for (size_t i = 0u; i < cloud->size(); ++i) {
    if (remove_point_inside) {
      if ((pow(cloud->points[i].x - center.x, 2.0)
          + pow(cloud->points[i].y - center.y, 2.0)) >= radius_squared ||
          abs(cloud->points[i].z - center.z) >= height_halved_m) {
        filtered_cloud.points.push_back(cloud->points[i]);
      }
    } else {
      if ((pow(cloud->points[i].x - center.x, 2.0)
          + pow(cloud->points[i].y - center.y, 2.0)) <= radius_squared &&
          abs(cloud->points[i].z - center.z) <= height_halved_m) {
        filtered_cloud.points.push_back(cloud->points[i]);
      }
    }
  }

  filtered_cloud.width = 1;
  filtered_cloud.height = filtered_cloud.points.size();

  *cloud = filtered_cloud;
}

static SE3 fromApproximateTransformationMatrix(const Eigen::Matrix4f& matrix) {
  SE3::Rotation rot = SE3::Rotation::fromApproximateRotationMatrix(
      matrix.cast<double>().topLeftCorner<3,3>().eval());
  SE3::Position pos = matrix.cast<double>().topRightCorner<3,1>().eval();
  return SE3(rot, pos);
}

static PclPoint laserSlamPoseToPclPoint(const laser_slam::Pose& pose) {
  PclPoint point;
  point.x = pose.T_w.getPosition()(0);
  point.y = pose.T_w.getPosition()(1);
  point.z = pose.T_w.getPosition()(2);
  return point;
}

static PclPoint se3ToPclPoint(const laser_slam::SE3& transform) {
  PclPoint point;
  point.x = transform.getPosition()(0);
  point.y = transform.getPosition()(1);
  point.z = transform.getPosition()(2);
  return point;
}

static double distanceBetweenTwoSE3(const SE3& pose1, const SE3& pose2) {
  return std::sqrt(
      (pose1.getPosition()(0) - pose2.getPosition()(0)) *
      (pose1.getPosition()(0) - pose2.getPosition()(0)) +
      (pose1.getPosition()(1) - pose2.getPosition()(1)) *
      (pose1.getPosition()(1) - pose2.getPosition()(1)) +
      (pose1.getPosition()(2) - pose2.getPosition()(2)) *
      (pose1.getPosition()(2) - pose2.getPosition()(2)));
}

static void transformPointCloud(const SE3& transform, PointICloud* point_cloud) {
  CHECK_NOTNULL(point_cloud);
  const Eigen::Matrix4f transform_matrix = transform.getTransformationMatrix().cast<float>();
  pcl::transformPointCloud(*point_cloud, *point_cloud, transform_matrix);
}

static void transformPclPoint(const SE3& transform, PclPoint* point) {
  CHECK_NOTNULL(point);
  Eigen::Matrix<double, 3, 1> eigen_point;
  eigen_point << point->x, point->y, point->z;
  eigen_point = transform.transform(eigen_point);
  point->x = eigen_point(0);
  point->y = eigen_point(1);
  point->z = eigen_point(2);
}

static laser_slam::Time findMostOccuringTime(const std::vector<laser_slam::Time>& times) {
  CHECK(!times.empty());
  std::map<laser_slam::Time, unsigned int> counts;
  for (const auto& time: times) {
    counts[time]++;
  }
  unsigned int max_count = 0u;
  laser_slam::Time most_occuring_timestamp = 0u;
  for (const auto& count: counts) {
    if (count.second > max_count) {
      max_count = count.second;
      most_occuring_timestamp = count.first;
    }
  }
  return most_occuring_timestamp;
}

static void applyRandomFilterToCloud(double ratio_of_points_to_keep,
                                     PointICloud* point_cloud) {
  if (ratio_of_points_to_keep != 1.0) {
    CHECK_NOTNULL(point_cloud);
    PointICloud filtered_cloud;
    // Manual filtering as pcl::RandomSample seems to be broken.
    for (size_t i = 0u; i < point_cloud->size(); ++i) {
      if (double(std::rand()) / double(RAND_MAX) < ratio_of_points_to_keep) {
        filtered_cloud.points.push_back(point_cloud->points[i]);
      }
    }

    filtered_cloud.width = 1;
    filtered_cloud.height = filtered_cloud.points.size();
    LOG(INFO) << "Filtering cloud from " << point_cloud->points.size() <<
        " to " << filtered_cloud.points.size() << " points.";
    *point_cloud = filtered_cloud;
  }
}

} // namespace segmatch

#endif // SEGMATCH_COMMON_HPP_
