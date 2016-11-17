cmake_minimum_required(VERSION 2.8.3)
project(laser_mapper)

add_definitions(-std=c++11)

find_package(catkin_simple 0.1.0 REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    genmsg
    message_generation
    tf
    rosbag
    visualization_msgs
)

catkin_simple(ALL_DEPS_REQUIRED)


cs_add_library(${PROJECT_NAME} 
    src/laser_mapper.cpp
)

add_executable(laser_mapper_node src/laser_mapper_node.cpp)
target_link_libraries(laser_mapper_node ${catkin_LIBRARIES} ${PROJECT_NAME})

find_package(Boost REQUIRED COMPONENTS system thread)

#add_doxygen(NOT_AUTOMATIC)

cs_install()
cs_export()

