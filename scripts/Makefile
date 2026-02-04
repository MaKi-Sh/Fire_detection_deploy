# Makefile for Fire Detection System with TensorRT and IR Camera
# This Makefile builds the main application and test utilities

# ============================= CONFIGURATION =============================
# Compiler and tools
CXX := g++
NVCC := nvcc
CUDA_ARCH := -gencode arch=compute_75,code=sm_75

# Project structure
PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BIN_DIR := $(PROJECT_DIR)bin
OBJ_DIR := $(PROJECT_DIR)obj
SRC_DIR := $(PROJECT_DIR)
INSTALL_PREFIX := /usr/local

# Source files
MAIN_SRC := $(SRC_DIR)main.cpp
TEST_SRC := $(SRC_DIR)test_deploy.cpp
DEPLOY_SRC := $(SRC_DIR)tensorRT_deploy.cpp
IR_SRC := $(SRC_DIR)IR_camera_detection.cpp

# Object files
MAIN_OBJ := $(OBJ_DIR)/main.o
TEST_OBJ := $(OBJ_DIR)/test_deploy.o
DEPLOY_OBJ := $(OBJ_DIR)/tensorRT_deploy.o
IR_OBJ := $(OBJ_DIR)/IR_camera_detection.o

# Binaries
MAIN_BIN := $(BIN_DIR)/fire_detection
TEST_BIN := $(BIN_DIR)/test_deploy

# Model path
ENGINE_PATH := /media/nvidia/0051-D5A7/yolo11n.engine

# ============================= COMPILER FLAGS =============================
# CUDA settings
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE := -I$(CUDA_PATH)/include
CUDA_LIB := -L$(CUDA_PATH)/lib64

# OpenCV settings
# Detect OpenCV location and use pkg-config if available
OPENCV_INCLUDE := -I/usr/include/opencv4 -I/usr/include
OPENCV_LIB := -L/usr/lib -L/usr/local/lib \
              -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
              -lopencv_imgcodecs -lopencv_dnn

# TensorRT settings
TENSORRT_PATH := /usr/local/tensorrt
TENSORRT_INCLUDE := -I$(TENSORRT_PATH)/include
TENSORRT_LIB := -L$(TENSORRT_PATH)/lib \
                -lnvinfer -lnvinfer_plugin -lcudart

# C++ compiler flags
CXX_FLAGS := -std=c++17 -Wall -Wextra -O3 -fPIC -pthread
DEBUG_FLAGS := -g -O0 -DDEBUG

# All include directories
INCLUDES := $(OPENCV_INCLUDE) $(CUDA_INCLUDE) $(TENSORRT_INCLUDE)

# All library directories and libraries
LIBS := $(OPENCV_LIB) $(CUDA_LIB) $(TENSORRT_LIB) -lcudart -lcuda

# ============================= TARGETS =============================
.PHONY: all main test clean help info create-dirs

# Default target (only builds main application)
all: create-dirs $(MAIN_BIN)

# Create necessary directories
create-dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)
	@echo "Created directories: $(BIN_DIR) $(OBJ_DIR)"

# Main application
$(MAIN_BIN): $(MAIN_OBJ)
	@echo "Linking main application..."
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Built: $(MAIN_BIN)"

# Test application
$(TEST_BIN): $(TEST_OBJ)
	@echo "Linking test application..."
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Built: $(TEST_BIN)"

# Object files compilation
$(OBJ_DIR)/main.o: $(MAIN_SRC)
	@echo "Compiling main.cpp..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_deploy.o: $(TEST_SRC)
	@echo "Compiling test_deploy.cpp..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Note: tensorRT_deploy.cpp and IR_camera_detection.cpp are included in main.cpp
# No separate object files needed since they use #include

# ============================= UTILITY TARGETS =============================

# Display configuration and environment info
info:
	@echo "=== Fire Detection System Build Configuration ==="
	@echo "C++ Compiler:       $(CXX)"
	@echo "CUDA Compiler:      $(NVCC)"
	@echo "Compiler Flags:     $(CXX_FLAGS)"
	@echo "Include Dirs:       $(INCLUDES)"
	@echo "Library Dirs:       $(LIBS)"
	@echo ""
	@echo "Source Directory:   $(SRC_DIR)"
	@echo "Output Directory:   $(BIN_DIR)"
	@echo "Object Directory:   $(OBJ_DIR)"
	@echo ""
	@echo "Main Binary:        $(MAIN_BIN)"
	@echo "Test Binary:        $(TEST_BIN)"
	@echo "Engine Path:        $(ENGINE_PATH)"
	@echo "=============================================="

# Run main application
run: $(MAIN_BIN)
	@echo "Running Fire Detection System..."
	@if [ ! -f "$(ENGINE_PATH)" ]; then \
		echo "WARNING: Engine file not found at $(ENGINE_PATH)"; \
		echo "Please update ENGINE_PATH in Makefile or check the path."; \
		exit 1; \
	fi
	$(MAIN_BIN)

# Run test application
test: $(TEST_BIN)
	@echo "Running test application..."
	$(TEST_BIN)

# Debug build
debug: CXX_FLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "✓ Debug build complete"

# Install binaries to system
install: all
	@echo "Installing to $(INSTALL_PREFIX)..."
	@mkdir -p $(INSTALL_PREFIX)/bin
	@cp $(MAIN_BIN) $(INSTALL_PREFIX)/bin/fire_detection
	@chmod 755 $(INSTALL_PREFIX)/bin/fire_detection
	@echo "✓ Installed to $(INSTALL_PREFIX)/bin/fire_detection"

# Uninstall binaries
uninstall:
	@rm -f $(INSTALL_PREFIX)/bin/fire_detection
	@echo "✓ Uninstalled from system"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "✓ Clean complete"

# Distclean - remove all generated files
distclean: clean
	@echo "Removing all generated files..."
	@echo "✓ Distclean complete"

# Help target
help:
	@echo "Fire Detection System - Makefile Help"
	@echo "======================================"
	@echo ""
	@echo "Available Targets:"
	@echo "  make              - Build main and test applications"
	@echo "  make main         - Build only main application"
	@echo "  make test         - Build and run test application"
	@echo "  make run          - Build and run main application"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make distclean    - Remove all generated files"
	@echo "  make install      - Install binaries to $(INSTALL_PREFIX)/bin"
	@echo "  make uninstall    - Remove installed binaries"
	@echo "  make info         - Display build configuration"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build all"
	@echo "  make run          # Build and run main application"
	@echo "  make debug        # Build with debug symbols"
	@echo "  make clean        # Clean build artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit the following variables at the top of Makefile:"
	@echo "  - CUDA_PATH:      CUDA installation directory"
	@echo "  - TENSORRT_PATH:  TensorRT installation directory"
	@echo "  - ENGINE_PATH:    Path to YOLO engine file"
	@echo "  - INSTALL_PREFIX: Installation prefix (default: /usr/local)"
	@echo ""

# ============================= BUILD DEPENDENCIES =============================
# Ensure proper dependency tracking
.PHONY: $(OBJ_DIR)/%.o

# Print variables for debugging
print-%:
	@echo "$* = $($*)"
