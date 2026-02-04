# Fire Detection System - Makefile Usage Guide

## Overview
A comprehensive Makefile has been created to build and manage the Fire Detection System with TensorRT and IR Camera support.

## What the Makefile Does
- Compiles `main.cpp` with embedded `tensorRT_deploy.cpp` and `IR_camera_detection.cpp`
- Builds a test application from `test_deploy.cpp`
- Manages object files and binaries in organized directories
- Provides useful development targets (debug, clean, install, etc.)

## Build Configuration
The Makefile automatically detects and configures:
- **OpenCV**: `/usr/include/opencv4`
- **CUDA**: `/usr/local/cuda`
- **TensorRT**: `/usr/local/tensorrt`
- **Compiler**: g++ with C++17 standard

## Quick Start

### Build all binaries
```bash
make
```

### Build and run main application
```bash
make run
```

### Build with debug symbols
```bash
make debug
```

### Run tests
```bash
make test
```

### Clean build artifacts
```bash
make clean
```

## Available Targets

| Target | Description |
|--------|-------------|
| `make` | Build main and test applications |
| `make main` | Build only main application |
| `make test` | Build and run test application |
| `make run` | Build and run main application |
| `make debug` | Build with debug symbols and no optimization |
| `make clean` | Remove all object files and binaries |
| `make distclean` | Remove all generated files |
| `make install` | Install binaries to `/usr/local/bin` |
| `make uninstall` | Remove installed binaries from system |
| `make info` | Display build configuration and paths |
| `make help` | Show this help message |

## Configuration

### Directory Structure
```
scripts/
├── Makefile              # This build file
├── main.cpp              # Main application
├── tensorRT_deploy.cpp   # TensorRT inference implementation
├── IR_camera_detection.cpp # IR thermal camera processing
├── test_deploy.cpp       # Test application
├── bin/                  # Output binaries (created by make)
└── obj/                  # Object files (created by make)
```

### Customizing Paths

Edit the Makefile variables if your system has different paths:

```makefile
# CUDA installation
CUDA_PATH := /usr/local/cuda

# TensorRT installation
TENSORRT_PATH := /usr/local/tensorrt

# Path to YOLO engine file
ENGINE_PATH := /media/nvidia/0051-D5A7/yolo11n.engine

# Installation prefix
INSTALL_PREFIX := /usr/local
```

## Output Files

After building:
- **Main binary**: `scripts/bin/fire_detection`
- **Test binary**: `scripts/bin/test_deploy`
- **Object files**: `scripts/obj/`

## Compilation Flags

| Flag | Purpose |
|------|---------|
| `-std=c++17` | Use C++17 standard |
| `-O3` | Maximum optimization |
| `-Wall -Wextra` | Show warnings |
| `-fPIC` | Position independent code |
| `-pthread` | Thread support |

Debug builds add `-g` (debug symbols) and `-O0` (no optimization).

## Dependencies

### Required Libraries
- OpenCV 4.x (core, imgproc, highgui, videoio, imgcodecs, dnn)
- CUDA Runtime
- TensorRT (NvInfer, NvInfer_plugin)

### System Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or later
- TensorRT SDK
- OpenCV 4.x

## Troubleshooting

### OpenCV not found
If you get "opencv2/opencv.hpp: No such file or directory":
```bash
# Install OpenCV development headers
sudo apt-get install libopencv-dev

# Or update the OPENCV_INCLUDE path in Makefile
OPENCV_INCLUDE := -I/path/to/opencv/include/opencv4
```

### TensorRT not found
```bash
# Update TensorRT path in Makefile
TENSORRT_PATH := /path/to/tensorrt

# Ensure TensorRT lib is in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH
```

### CUDA not found
```bash
# Ensure CUDA is installed
which nvcc

# Update CUDA path if needed
CUDA_PATH := /usr/local/cuda
```

## Environment Variables

Before running the application, ensure:
```bash
# Add TensorRT libraries to library path
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH

# Run the application
./scripts/bin/fire_detection
```

## Performance Tips

1. **Use Release Build** (default):
   ```bash
   make clean && make
   ```

2. **Profile for Optimization**:
   ```bash
   make debug
   # Then use gdb or valgrind for profiling
   ```

3. **Check GPU Memory**:
   ```bash
   nvidia-smi
   ```

## Integration with CI/CD

The Makefile can be integrated into CI/CD pipelines:

```bash
# Build in CI
make clean
make info
make

# Run tests
make test

# Check for compiler warnings
make CXX_FLAGS+="-Werror"
```

## Notes

- The Makefile uses header-only includes for `tensorRT_deploy.cpp` and `IR_camera_detection.cpp` (via `#include` in main.cpp)
- No separate compilation of these files is needed
- The `engine_path` (yolo11n.engine) must exist for the application to run
- RGB camera ID: 10, IR camera ID: 11 (configured in main.cpp)
