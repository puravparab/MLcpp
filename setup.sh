#!/bin/bash

# Prompt the user for the directory
echo "Enter the example directory:"
read -r example_dir

# Create a build directory
mkdir -p "examples/$example_dir/build"

# Navigate to the build directory
cd "examples/$example_dir/build" || exit

# Run CMake to generate Makefiles
cmake ..

# Run make to build the project
make

# Run example
./mlcpp

# Return to the original directory
cd ..