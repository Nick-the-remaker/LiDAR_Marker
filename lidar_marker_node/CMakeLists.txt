cmake_minimum_required(VERSION 3.0.2)
project(lidar_marker)

set(CMAKE_CXX_FLAGS "-std=c++14")

# set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  message_generation 
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  pcl_conversions
  pcl_ros
  # livox_ros_driver2
)

find_package(OpenCV 4 REQUIRED )

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(process_pcd src/process_pcd.cpp)


target_link_libraries(process_pcd 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  pcl_visualization
  pcl_common
  pcl_io
  pcl_features
  pcl_visualization
  pcl_kdtree
  pcl_search
)



