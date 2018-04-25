cmake_minimum_required(VERSION 2.8.3)
project(segmapper)

add_definitions(-std=c++11 -DBENCHMARK_ENABLE)

find_package(catkin_simple 0.1.0 REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME} 
    src/segmapper.cpp
)

cs_add_executable(segmapper_node src/segmapper_node.cpp)
target_link_libraries(segmapper_node ${PROJECT_NAME})

cs_install()
cs_export()

