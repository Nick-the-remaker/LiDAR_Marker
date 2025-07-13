# LiDAR Marker: An Intensity-Based Region Growing Fiducial Marker System for LiDAR

## Overview
This C++ code processes point cloud data to detect LiDAR Markers and analyzes their geometric properties.

## Dependencies
- **Libraries**:
  - PCL (Point Cloud Library) 1.8+
  - Eigen3
  - Boost Thread
- **Data Formats**: PCD (Point Cloud Data)

## Compilation
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile
make -j4
