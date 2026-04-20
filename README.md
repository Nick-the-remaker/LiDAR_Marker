# LiDAR Marker: An Intensity-Based Region Growing Fiducial Marker System for LiDAR
<!-- <img src="lidar_marker_node/images/cover.gif" width = "820" alt="center" align=center /> -->

<div align="center">
  <img src="lidar_marker_node/images/cover.gif" alt="cover" width="820">
</div>

## Overview
This repository is for the intensity-based region growing fiducial LiDAR Marker system.



<img src="lidar_marker_node/images/pipeline.png" width = "820" height = "800" alt="pipeline" align=center />

## Dependencies
- **Libraries**:
  - PCL (Point Cloud Library) 1.8+
  - Eigen3
  - Boost Thread
  - OpenCV


## Compilation
```bash
cd catkin_ws/src
# Then clone this repository.

cd ..
catkin_make

# Run the code
source ./devel/setup.bash
rosrun lidar_marker process_pcd
```
