# LiDAR Marker: An Intensity-Based Region Growing Fiducial Marker System for LiDAR

## Overview
This repository is for the intensity-based region growing fiducial LiDAR Marker system.

<img src="lidar_marker_node/images/pipeline.png" width = "820" height = "800" alt="pipeline" align=center />

## Dependencies
- **Libraries**:
  - PCL (Point Cloud Library) 1.8+
  - Eigen3
  - Boost Thread
  - OpenCV

## Dataset
You can find the data used in the paper through this link: [Dataset](https://1drv.ms/f/c/ec46e108ef532789/EsdJFNEZ5dlJk7F3alWmph8Bl6WV5qc_wg0h53TBq-ctHw?e=lg9rL5).

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
