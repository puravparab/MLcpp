#!/bin/bash

# Create a build directory
mkdir -p build

# Navigate to the build directory
cd build

# Run CMake to generate Makefiles
cmake ..

# Run make to build the project
make

# Return to the original directory
cd ..