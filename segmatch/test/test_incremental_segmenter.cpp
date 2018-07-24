#include <glog/logging.h>
#include <gtest/gtest.h>
#include <segmatch/segmenters/region_growing_policy.hpp>

#include "segmatch/points_neighbors_providers/kdtree_points_neighbors_provider.hpp"
#include "segmatch/segmented_cloud.hpp"
#include "segmatch/segmenters/incremental_segmenter.hpp"

using namespace segmatch;

// Initialize common objects needed by multiple tests.
class IncrementalEuclideanSegmenterTest : public ::testing::Test {
 protected:
  typedef MapPoint ClusteredPointT;
  typedef IncrementalSegmenter<ClusteredPointT, EuclideanDistance> Segmenter;
  typedef std::vector<float> Indices;

  Segmenter segmenter_;
  SegmentedCloud segmented_cloud_;
  KdTreePointsNeighborsProvider<ClusteredPointT> kdtree_;

  IncrementalEuclideanSegmenterTest()
    : segmenter_(createParameters(2.0f, 1000, 2)) {
  }

  void SetUp() override {
    segmented_cloud_.resetSegmentIdCounter();
  }

  void TearDown() override {
  }

  // Shortcut for setting euclidean segmenter related parameters.
  static SegmenterParameters createParameters(const double radius_for_growing,
                                              const int max_cluster_size,
                                              const int min_cluster_size) {
    SegmenterParameters parameters;
    parameters.radius_for_growing = radius_for_growing;
    parameters.max_cluster_size = max_cluster_size;
    parameters.min_cluster_size = min_cluster_size;
    return parameters;
  }

  // Compensate PCL's lack of a proper PointXYZI constructor.
  static ClusteredPointT createClusteredPointT(const float x, const float y,
                                               const float z,
                                               const uint32_t cluster_id = 0u) {
    ClusteredPointT point;
    point.x = x;
    point.y = y;
    point.z = z;
    point.ed_cluster_id = cluster_id;
    return point;
  }

  // Compensate PCL's lack of a PointCloud initializer list constructor.
  static Segmenter::ClusteredCloud createClusteredCloud(
      std::initializer_list<std::array<float, 4>> points) {
    Segmenter::ClusteredCloud cloud;
    cloud.reserve(points.size());
    for(const auto& p : points) {
       cloud.push_back(createClusteredPointT(p[0], p[1], p[2], static_cast<uint32_t>(p[3])));
    }
    return cloud;
  }

  static Indices getClusterIndices(const Segmenter::ClusteredCloud& cloud) {
    Indices cluster_indices;
    cluster_indices.reserve(cloud.size());
    for (const auto& point : cloud) {
      cluster_indices.push_back(point.ed_cluster_id);
    }
    return cluster_indices;
  }

  void verifySegmentationResult(
      const Segmenter::ClusteredCloud& cloud,
      const std::vector<Id>& segments,
      const std::vector<std::pair<Id, Id>> renamed_segments,
      const Indices& expected_clusters,
      const std::vector<Id>& expected_segments,
      const std::vector<size_t>& expected_segment_sizes,
      const std::vector<std::pair<Id, Id>> expected_renamed_segments) const {
    size_t valid_segments_count = 0;
    for (const auto& segment_id : expected_segments) {
      if (segment_id > 0) valid_segments_count++;
    }

    EXPECT_EQ(expected_clusters, getClusterIndices(cloud));
    EXPECT_EQ(expected_segments, segments);
    EXPECT_EQ(valid_segments_count,
              segmented_cloud_.getNumberOfValidSegments());

    for (int i = 0; i < expected_segments.size(); i++) {
      if (expected_segments[i] > 0) {
        Segment segment;
        ASSERT_TRUE(segmented_cloud_.findValidSegmentById(expected_segments[i],
                                                          &segment));
        ASSERT_EQ(expected_segment_sizes[i], segment.getLastView().point_cloud.size());
      }
    }

    EXPECT_EQ(expected_renamed_segments, renamed_segments);
  }
};

TEST_F(IncrementalEuclideanSegmenterTest, test_single_cluster) {

  // Arrange
  Segmenter::ClusteredCloud cloud = createClusteredCloud({
    {   0.0f,  0.0f,  0.0f },  // 1
    { -1.99f,  0.0f,  0.0f },  // 1
    {   1.0f,  0.4f, -0.7f },  // 1
    {   0.3f,  0.7f,  0.2f },  // 1
    {   0.6f, -0.2f,  0.4f }   // 1
  });
  std::vector<Id> segments { };
  std::vector<std::pair<Id, Id>> renamed_segments;
  kdtree_.update(typename MapCloud::Ptr(&cloud, [](MapCloud* ptr) {}), { });

  // Act
  segmenter_.segment({ }, { }, cloud, kdtree_, segmented_cloud_, segments, renamed_segments);

  // Assert
  verifySegmentationResult(
      cloud, segments, renamed_segments,
      { 1.0f, 1.0, 1.0f, 1.0f, 1.0f },  // Expected clusters
      { kNoId, 1 },                     // Expected segments
      { 0, 5 },                         // Expected segments sizes
      { });                             // Expected renamed segments
}

TEST_F(IncrementalEuclideanSegmenterTest, test_multiple_clusters) {

  // Arrange
  Segmenter::ClusteredCloud cloud = createClusteredCloud({
    {   0.0f,  0.0f,  0.0f }, // 1
    {  10.4f,  9.4f,  9.2f }, // 2
    {   0.4f,  0.4f, -0.2f }, // 1
    { -50.4f,  0.4f,  0.2f }, // 3
    {  -0.3f, -0.8f,  0.3f }, // 1
    {  10.2f, 10.3f,  9.3f }  // 2
  });
  std::vector<Id> segments { };
  std::vector<std::pair<Id, Id>> renamed_segments;
  kdtree_.update(typename MapCloud::Ptr(&cloud, [](MapCloud* ptr) {}), { });

  // Act
  segmenter_.segment({ }, { }, cloud, kdtree_, segmented_cloud_, segments, renamed_segments);

  // Assert
  verifySegmentationResult(
      cloud, segments, renamed_segments,
      { 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 2.0f },  // Expected clusters
      { kNoId, 1, 2 , kNoId },                 // Expected segments
      { 0, 3, 2, 0 },                          // Expected segments sizes
      { });                                    // Expected renamed segments
}

TEST_F(IncrementalEuclideanSegmenterTest, test_incremental_clustering) {

  // Arrange
  Segmenter::ClusteredCloud cloud = createClusteredCloud({
    {  -5.0f,  2.0f, -4.2f, 6.0f },     // Cluster 6, expected reassign to 3
    { -73.0f,  3.0f,  0.2f, 2.0f },     // Cluster 2, expected reassign to 1
    {   1.4f, 16.3f, -4.2f, 0.0f },     // Expected join 3 and reassign to 2
    {   1.8f, -2.3f, 34.8f, 0.0f },     // Expected 4
    {  -4.8f,  2.5f, -3.6f, 6.0f },     // Cluster 6, expected reassign to 3
    {  -5.4f,  4.2f, -4.4f, 0.0f },     // Expected join 6 and reassign to 3
    {  -5.4f,  2.3f, -4.4f, 6.0f },     // Cluster 6, expected reassign to 3
    {   2.0f, 15.0f, -3.4f, 3.0f }      // Cluster 3, expected reassign to 2
  });
  std::vector<Id> segments { kNoId, kNoId, kNoId, kNoId, kNoId, kNoId, 1 };
  std::vector<std::pair<Id, Id>> renamed_segments;
  segmented_cloud_.resetSegmentIdCounter(2); // 1 is the last segment IDs used.
  kdtree_.update(typename MapCloud::Ptr(&cloud, [](MapCloud* ptr) {}), { });

  // Act
  segmenter_.segment({ }, { }, cloud, kdtree_, segmented_cloud_, segments, renamed_segments);

  // Assert
  verifySegmentationResult(
      cloud, segments, renamed_segments,
      { 3.0f, 1.0f, 2.0f, 4.0f, 3.0f, 3.0f, 3.0f, 2.0f }, // Expected clusters
      { kNoId, kNoId, 2, 1, kNoId },                      // Expected segments
      { 0, 0, 2, 4, 0 },                                  // Expected segments sizes
      { });                                               // Expected renamed segments
}


TEST_F(IncrementalEuclideanSegmenterTest, test_join_clusters) {

  // Arrange
  Segmenter::ClusteredCloud cloud = createClusteredCloud({
    {  0.0f,  0.0f,  0.0f, 1.0f },   // 1
    {  1.8f, -2.3f, 34.8f, 2.0f },   // 2
    {  1.5f,  0.2f, -0.1f, 0.0f },   // Expect join 1 and 3
    {  3.2f,  0.5f, -0.8f, 3.0f },   // 3
    { -0.4f, -0.3f, -1.2f, 1.0f },   // 1
    {  3.0f,  0.0f,  0.0f, 3.0f },   // 3
  });
  std::vector<Id> segments { kNoId, 1, kNoId, 2 };
  std::vector<std::pair<Id, Id>> renamed_segments;
  segmented_cloud_.resetSegmentIdCounter(3); // 2 is the last segment IDs used.
  kdtree_.update(typename MapCloud::Ptr(&cloud, [](MapCloud* ptr) {}), { });

  // Act
  segmenter_.segment({ }, { }, cloud, kdtree_, segmented_cloud_, segments, renamed_segments);

  // Assert
  verifySegmentationResult(
      cloud, segments, renamed_segments,
      { 1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f },  // Expected clusters
      { kNoId, 1, kNoId },                     // Expected segments
      { 0, 5, 0 },                             // Expected segments sizes
      { { 2, 1 } });                           // Expected renamed segments
}
