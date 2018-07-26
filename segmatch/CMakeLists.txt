cmake_minimum_required(VERSION 2.8.3)
project(segmatch)

find_package(catkin_simple 0.1.0 REQUIRED)
catkin_simple()

add_definitions(-std=c++11 -DBENCHMARK_ENABLE)

cs_add_library(${PROJECT_NAME} 
  src/database.cpp
  src/descriptors/cnn.cpp
  src/descriptors/descriptors.cpp
  src/descriptors/eigenvalue_based.cpp
  src/descriptors/ensemble_shape_functions.cpp
  src/dynamic_voxel_grid.cpp
  src/features.cpp
  src/local_map.cpp
  src/normal_estimators/incremental_normal_estimator.cpp
  src/normal_estimators/normal_estimator.cpp
  src/normal_estimators/simple_normal_estimator.cpp
  src/opencv_random_forest.cpp
  src/points_neighbors_providers/kdtree_points_neighbors_provider.cpp
  src/points_neighbors_providers/octree_points_neighbors_provider.cpp
  src/recognizers/correspondence_recognizer_factory.cpp
  src/recognizers/geometric_consistency_recognizer.cpp
  src/recognizers/graph_based_geometric_consistency_recognizer.cpp
  src/recognizers/incremental_geometric_consistency_recognizer.cpp
  src/recognizers/partitioned_geometric_consistency_recognizer.cpp
  src/rviz_utilities.cpp
  src/segmatch.cpp
  src/segmented_cloud.cpp
  src/segmenters/euclidean_segmenter.cpp
  src/segmenters/incremental_segmenter.cpp
  src/segmenters/segmenter_factory.cpp
  src/segmenters/smoothness_constraints_segmenter.cpp
)
target_link_libraries(${PROJECT_NAME})

find_package(Boost REQUIRED COMPONENTS system thread)

catkin_add_gtest(${PROJECT_NAME}_tests 
  test/test_main.cpp
  test/test_dynamic_voxel_grid.cpp
  test/test_geometric_consistency_recognizer.cpp
  test/test_graph_utilities.cpp
  test/test_incremental_segmenter.cpp
  test/test_incremental_geometric_consistency_recognizer.cpp
  test/test_incremental_normal_estimator.cpp
  test/test_matches_partitioner.cpp
  test/test_partitioned_geometric_consistency_recognizer.cpp
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

add_doxygen(REQUIRED)

cs_install()
cs_export()
