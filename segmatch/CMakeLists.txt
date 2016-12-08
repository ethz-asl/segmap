cmake_minimum_required(VERSION 2.8.3)
project(segmatch)

find_package(catkin_simple 0.1.0 REQUIRED)
find_package(OpenCV 2 REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} 
  src/descriptors/descriptors.cpp
  src/descriptors/eigenvalue_based.cpp
  src/descriptors/autoencoder.cpp
  src/descriptors/ensemble_shape_functions.cpp
  src/features.cpp
  src/database.cpp
  src/opencv_random_forest.cpp
  src/segmatch.cpp
  src/segmented_cloud.cpp
  src/segmenters/euclidean_segmenter.cpp
  src/segmenters/region_growing_segmenter.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

find_package(Boost REQUIRED COMPONENTS system thread)

catkin_add_gtest(${PROJECT_NAME}_tests 
  test/test_main.cpp
  test/test_empty.cpp
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

add_doxygen(REQUIRED)

cs_install()
cs_export()
