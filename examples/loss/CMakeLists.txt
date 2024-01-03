cmake_minimum_required(VERSION 3.28.1)
project(mlcpp)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the path to the source files
set(SOURCES
  "../../src/dataset/dataset.cpp"
  "../../src/dataset/dataloader.cpp"
  "../../src/normalization/norm.cpp"
  "../../src/loss/loss.cpp"
  "../../examples/loss/main.cpp"
)

# specify the path to the test files
file(GLOB TEST_SOURCES "../../test/*.cpp")

# add the executable
add_executable(mlcpp ${SOURCES} ${TEST_SOURCES})

# Set the path to Eigen headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/eigen)