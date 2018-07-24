#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "segmatch/dynamic_voxel_grid.hpp"
#include "segmatch/impl/dynamic_voxel_grid.hpp"

using namespace segmatch;

namespace pcl {
  // Custom equality operator for testing.
  template<typename PointT>
  bool operator==(const PointT& p1, const PointT& p2) {
    return std::abs(p1.x - p2.x) < 0.001
        && std::abs(p1.y - p2.y) < 0.001
        && std::abs(p1.z - p2.z) < 0.001;
  }
}

// Initialize common objects needed by multiple tests.
class DynamicVoxelGridTest : public ::testing::Test {
 protected:
  typedef pcl::PointXYZ InputPointT;
  typedef pcl::PointXYZI VoxelPointT;
  typedef uint64_t IndexT;
  typedef uint8_t SmallIndexT;
  typedef DynamicVoxelGrid<
      InputPointT,
      VoxelPointT,
      IndexT, 20, 20, 20> VoxelGrid;
  typedef DynamicVoxelGrid<
      InputPointT,
      VoxelPointT,
      SmallIndexT, 3, 3, 2> SmallVoxelGrid;

  const float resolution_ = 1.0f;
  InputPointT origin_;
  SmallVoxelGrid::InputCloud small_insert_1_;
  SmallVoxelGrid::InputCloud small_insert_2_;
  SmallVoxelGrid::InputCloud small_insert_3_;
  SmallVoxelGrid::InputCloud big_insert_1_;
  SmallVoxelGrid::InputCloud big_insert_2_;

  VoxelGrid grid_;
  SmallVoxelGrid small_grid_;
  SmallVoxelGrid shifted_grid_;

  DynamicVoxelGridTest()
    : origin_(1.0f, -2.0f, 0.0f)
    , grid_(resolution_, 3)
    , small_grid_(resolution_, 2)
    , shifted_grid_(resolution_, 2, origin_) {
  }

  void SetUp() override {
    small_insert_1_.push_back(InputPointT(0.2f, 1.5f, 0.1f)); // Voxel 1
    small_insert_1_.push_back(InputPointT(0.8f, 0.7f, 0.9f)); // Voxel 2
    small_insert_1_.push_back(InputPointT(0.5f, 0.6f, 0.3f)); // Voxel 2
    small_insert_1_.push_back(InputPointT(1.3f, 0.1f, 1.6f)); // Voxel 3

    small_insert_2_.push_back(InputPointT(0.4f, 0.7f, 0.2f)); // Voxel 2
    small_insert_2_.push_back(InputPointT(1.3f, 0.3f, 1.6f)); // Voxel 3
    small_insert_2_.push_back(InputPointT(1.3f, 0.1f,-1.2f)); // Voxel 4

    small_insert_3_.push_back(InputPointT(0.4f, 0.2f, 0.2f)); // Voxel 2

    big_insert_1_.push_back(InputPointT(3.5f, -6.2f, 0.7f)); // Voxel 1
    big_insert_1_.push_back(InputPointT(10.0f, 3.2f, 4.7f)); // Voxel 2
    big_insert_1_.push_back(InputPointT(-34.5f, 37896.1f, 234.2f)); // Voxel 3
    big_insert_1_.push_back(InputPointT(-34.5f, 37895.9f, 234.2f)); // Voxel 4
    big_insert_1_.push_back(InputPointT(0.5f, 0.2f, 0.7f)); // Voxel 5

    big_insert_1_.push_back(InputPointT(0.4f, 1.2f, 0.5f)); // Voxel 6
    big_insert_1_.push_back(InputPointT(-34.4f, 37896.1f, 234.5f)); // Voxel 3
    big_insert_1_.push_back(InputPointT(10.3f, 3.7f, 4.3f)); // Voxel 2
    big_insert_1_.push_back(InputPointT(10.3f, 3.7f, 4.3f)); // Voxel 2
    big_insert_1_.push_back(InputPointT(0.6f, 0.4f, 0.7f)); // Voxel 5

    big_insert_1_.push_back(InputPointT(0.1f, 0.4f, 0.7f)); // Voxel 5
    big_insert_1_.push_back(InputPointT(0.6f, 0.4f, 0.9f)); // Voxel 5
    big_insert_1_.push_back(InputPointT(-34.3f, 37895.9f, 234.2f)); // Voxel 4
    big_insert_1_.push_back(InputPointT(-34.5f, 37895.9f, 234.9f)); // Voxel 4
    big_insert_1_.push_back(InputPointT(0.4f, 1.2f, 0.9f)); // Voxel 6

    big_insert_2_.push_back(InputPointT(-34.1f, 37896.6f, 234.64f)); // Voxel 3
    big_insert_2_.push_back(InputPointT(0.5f, 0.2f, 0.7f)); // Voxel 5
    big_insert_2_.push_back(InputPointT(3.5f, -6.2f, 0.7f)); // Voxel 1
    big_insert_2_.push_back(InputPointT(-500.2f, 33.2f, 4.7f)); // Voxel 7
    big_insert_2_.push_back(InputPointT(-500.5f, 33.6f, 4.9f)); // Voxel 7
  }

  void TearDown() override {
  }

  VoxelPointT createVoxelPointT(const float x, const float y, const float z,
                               const float i = 0.0f) {
    VoxelPointT point(i);
    point.x = x;
    point.y = y;
    point.z = z;
    return point;
  }
};

TEST_F(DynamicVoxelGridTest, test_creation_is_empty) {
  EXPECT_EQ(grid_.getActiveCentroids().size(), 0);
  EXPECT_EQ(grid_.getInactiveCentroids().size(), 0);
}

TEST_F(DynamicVoxelGridTest, test_getIndexOf) {
  EXPECT_EQ(SmallIndexT(0x00),
            small_grid_.getIndexOf(InputPointT(-3.5f, -3.5f, -1.5f)));
  EXPECT_EQ(SmallIndexT(0xD6),
            small_grid_.getIndexOf(InputPointT( 2.2f, -1.5f,  1.8f)));
  EXPECT_EQ(SmallIndexT(0x27),
            small_grid_.getIndexOf(InputPointT( 3.0f,  0.0f, -2.0f)));
}

TEST_F(DynamicVoxelGridTest, test_getIndexOf_with_offset) {
  EXPECT_EQ(SmallIndexT(0x00),
            shifted_grid_.getIndexOf(InputPointT(-2.5f, -5.5f, -1.5f)));
  EXPECT_EQ(SmallIndexT(0xD6),
            shifted_grid_.getIndexOf(InputPointT( 3.2f, -3.5f,  1.8f)));
  EXPECT_EQ(SmallIndexT(0x27),
            shifted_grid_.getIndexOf(InputPointT( 4.0f, -2.0f, -2.0f)));
}

TEST_F(DynamicVoxelGridTest, test_getIndexOf_big_grid) {
  EXPECT_NE(
      grid_.getIndexOf(InputPointT(-34.5f ,37896.1f ,234.2f)),
      grid_.getIndexOf(InputPointT(-34.5f ,37895.9f ,234.2f)));
  EXPECT_EQ(
      grid_.getIndexOf(InputPointT(0.1f, 0.4f, 0.7f)),
      grid_.getIndexOf(InputPointT(0.6f ,0.4f ,0.7f)));
}

TEST_F(DynamicVoxelGridTest, test_one_insert) {
  auto created = shifted_grid_.insert(small_insert_1_);

  ASSERT_EQ(1, shifted_grid_.getActiveCentroids().size());
  EXPECT_EQ(2, shifted_grid_.getInactiveCentroids().size());
  EXPECT_EQ(createVoxelPointT(0.65f, 0.65f, 0.6f),
            shifted_grid_.getActiveCentroids()[0]);

  ASSERT_EQ(1, created.size());
  EXPECT_EQ(0, created[0]);
}

TEST_F(DynamicVoxelGridTest, test_voxel_fields_are_preserved) {
  shifted_grid_.insert(small_insert_1_);
  EXPECT_EQ(0.0f, shifted_grid_.getActiveCentroids()[0].intensity);
  shifted_grid_.getActiveCentroids()[0].intensity = 10.0f;
  EXPECT_EQ(10.0f, shifted_grid_.getActiveCentroids()[0].intensity);

  shifted_grid_.insert(small_insert_3_);
  EXPECT_EQ(10.0f, shifted_grid_.getActiveCentroids()[0].intensity);
}

TEST_F(DynamicVoxelGridTest, test_two_insert) {
  shifted_grid_.insert(small_insert_1_);
  auto created = shifted_grid_.insert(small_insert_2_);

  EXPECT_EQ(2, shifted_grid_.getActiveCentroids().size());
  EXPECT_EQ(2, shifted_grid_.getInactiveCentroids().size());

  ASSERT_EQ(1, created.size());
  EXPECT_EQ(1, created[0]);
}

TEST_F(DynamicVoxelGridTest, test_remove) {
  shifted_grid_.insert(small_insert_1_);
  shifted_grid_.insert(small_insert_2_);
  size_t num_active_voxels = shifted_grid_.getActiveCentroids().size();

  auto removed = shifted_grid_.removeIf([](const VoxelPointT& p){
    return p.x == 1.3f;
  });

  ASSERT_EQ(num_active_voxels, removed.size());
  EXPECT_EQ(std::vector<bool>({ false, true }), removed);
  EXPECT_EQ(1, shifted_grid_.getActiveCentroids().size());
  EXPECT_EQ(1, shifted_grid_.getInactiveCentroids().size());
}

TEST_F(DynamicVoxelGridTest, test_full) {
  grid_.insert(big_insert_1_);
  EXPECT_EQ(3, grid_.getActiveCentroids().size());
  EXPECT_EQ(3, grid_.getInactiveCentroids().size());

  auto removed = grid_.removeIf([this](const VoxelPointT& p){
    return grid_.getIndexOf(p) == grid_.getIndexOf(big_insert_1_[3])
        || grid_.getIndexOf(p) == grid_.getIndexOf(big_insert_1_[5]);
  });
  EXPECT_EQ(2, grid_.getActiveCentroids().size());
  EXPECT_EQ(2, grid_.getInactiveCentroids().size());
  EXPECT_EQ(3, removed.size());

  grid_.insert(big_insert_2_);
  EXPECT_EQ(3, grid_.getActiveCentroids().size());
  EXPECT_EQ(2, grid_.getInactiveCentroids().size());

  removed = grid_.removeIf([this](const VoxelPointT& p){
    return grid_.getIndexOf(p) == grid_.getIndexOf(big_insert_1_[1]);
  });
  EXPECT_EQ(2, grid_.getActiveCentroids().size());
  EXPECT_EQ(2, grid_.getInactiveCentroids().size());
  EXPECT_EQ(3, removed.size());
}

TEST_F(DynamicVoxelGridTest, test_clear) {
  // Arrange
  grid_.insert(big_insert_1_);
  ASSERT_LE(1, grid_.getActiveCentroids().size());
  ASSERT_LE(1, grid_.getInactiveCentroids().size());

  // Act
  grid_.clear();

  // Assert
  EXPECT_EQ(0, grid_.getActiveCentroids().size());
  EXPECT_EQ(0, grid_.getInactiveCentroids().size());
}

TEST_F(DynamicVoxelGridTest, test_transform) {
  // Arrange
  kindr::minimal::QuatTransformationTemplate<float> transformation(
      kindr::minimal::QuatTransformationTemplate<float>::Position(-1.4f, 1.3f, 0.0f),
      kindr::minimal::RotationQuaternionTemplate<float>(Eigen::Vector3f( 1.0f, 0.5f, -0.7f )));
  InputPointT original_point(1.2f, -2.3f, 1.9f);
  InputPointT transformed_point;
  transformed_point.getArray3fMap() = transformation.transform(original_point.getArray3fMap());
  IndexT original_point_index = shifted_grid_.getIndexOf(original_point);

  // Sanity check. If the original and transformed points have the same voxel index before
  // transforming the grid, the test is ineffective.
  ASSERT_NE(original_point_index, shifted_grid_.getIndexOf(transformed_point));

  // Act
  shifted_grid_.transform(transformation);
  IndexT transformed_point_index = shifted_grid_.getIndexOf(transformed_point);


  // Assert
  EXPECT_EQ(original_point_index, transformed_point_index);
}
