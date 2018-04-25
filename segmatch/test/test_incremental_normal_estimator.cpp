#include <glog/logging.h>
#include <gtest/gtest.h>

#include "segmatch/normal_estimators/incremental_normal_estimator.hpp"
#include "segmatch/points_neighbors_providers/kdtree_points_neighbors_provider.hpp"

using namespace segmatch;

// Initialize common objects needed by multiple tests.
class IncrementalNormalEstimatorTest : public ::testing::Test {
 protected:
  IncrementalNormalEstimator estimator_;
  MapCloud points_;
  MapCloud updated_points_;
  typename MapCloud::ConstPtr points_ptr_;
  typename MapCloud::ConstPtr updated_points_ptr_;
  KdTreePointsNeighborsProvider<MapPoint> kd_tree_;

  IncrementalNormalEstimatorTest()
    : estimator_(0.5f), points_ptr_(&points_, [](MapCloud const* ptr) {}),
      updated_points_ptr_(&updated_points_, [](MapCloud const* ptr) {}) {
  }

  void SetUp() override {
    points_ = createMapCloud({
      {   0.0f,  0.0f, 0.0f },
      {  0.25f,  0.0f, 0.0f },
      {   0.0f,  0.1f, 0.0f },
      { -0.25f,  0.0f, 0.0f },
      {   0.0f, -0.1f, 0.0f },
      {   1.0f,  1.0f, 0.0f },
    });

    updated_points_ = createMapCloud({
      {   0.0f,  0.0f,  0.0f },
      {  0.25f,  0.0f,  0.0f },
      {   0.0f,  0.1f,  0.0f },
      {   1.0f,  1.0f,  0.0f },
      {   0.0f,  0.0f,  0.4f },
      {   0.0f,  0.0f, -0.4f },
      {   1.2f,  1.0f,  0.0f },
      {   1.0f,  1.2f,  0.0f },
    });
  }

  void TearDown() override {
  }

  // Construct a MapPoint.
  static MapPoint createMapPoint(const float x, const float y, const float z) {
    MapPoint point;
    point.x = x;
    point.y = y;
    point.z = z;
    return point;
  }

  // Create a PointCloud with an initializer list.
  static MapCloud createMapCloud(std::initializer_list<std::array<float, 3>> points) {
    MapCloud cloud;
    cloud.reserve(points.size());
    for(const auto& p : points) {
       cloud.push_back(createMapPoint(p[0], p[1], p[2]));
    }
    return cloud;
  }
};

TEST_F(IncrementalNormalEstimatorTest, test_nan_normal) {

  // Arrange
  kd_tree_.update(points_ptr_);

  // Act
  estimator_.updateNormals(points_, { }, { 0, 1, 2, 3, 4, 5 }, kd_tree_);

  // Assert
  // This point has no neighbor, thus the normal cannot be estimated and is set to NaN.
  const PclNormal& nan_normal = estimator_.getNormals()[5];
  EXPECT_TRUE(std::isnan(nan_normal.normal_x) && std::isnan(nan_normal.normal_y) &&
              std::isnan(nan_normal.normal_z) && std::isnan(nan_normal.curvature));
}

TEST_F(IncrementalNormalEstimatorTest, test_normal) {

  // Arrange
  kd_tree_.update(points_ptr_);

  // Act
  estimator_.updateNormals(points_, { }, { 0, 1, 2, 3, 4, 5 }, kd_tree_);

  // Assert
  EXPECT_EQ(0.0f, estimator_.getNormals()[0].normal_x);
  EXPECT_EQ(0.0f, estimator_.getNormals()[0].normal_y);
  EXPECT_EQ(1.0f, estimator_.getNormals()[0].normal_z);
}

TEST_F(IncrementalNormalEstimatorTest, test_incremental) {

  // Arrange
  kd_tree_.update(points_ptr_);

  // Act
  estimator_.updateNormals(points_, { }, { 0, 1, 2, 3, 4, 5 }, kd_tree_);
  kd_tree_.update(updated_points_ptr_);
  estimator_.updateNormals(updated_points_, { 0, 1, 2, -1, -1, 3 }, { 4, 5, 6, 7 }, kd_tree_);

  // Assert
  EXPECT_EQ(0.0f, estimator_.getNormals()[0].normal_x);
  EXPECT_EQ(1.0f, estimator_.getNormals()[0].normal_y);
  EXPECT_EQ(0.0f, estimator_.getNormals()[0].normal_z);
}
