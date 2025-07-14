# LiDAR Marker: An Intensity-Based Region Growing Fiducial Marker System for LiDAR

## Overview
This C++ code processes point cloud data to detect LiDAR Markers and analyzes their geometric properties.

## Dependencies
- **Libraries**:
  - PCL (Point Cloud Library) 1.8+
  - Eigen3
  - Boost Thread
- **Data Formats**: PCD (Point Cloud Data)

## Dataset
You can find the data used in the paper through this link: [Dataset](https://1drv.ms/f/c/ec46e108ef532789/EsdJFNEZ5dlJk7F3alWmph8Bl6WV5qc_wg0h53TBq-ctHw?e=lg9rL5).

## Compilation
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile
make -j4
