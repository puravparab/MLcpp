cmake_minimum_required(VERSION 3.28.1)
project(MLcpp)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the path to the header files
include_directories(include)
include_directories(libs)

# specify the path to the source files
file(GLOB_RECURSE SOURCES "*.cpp")

# specify the path to the test files
file(GLOB TEST_SOURCES "test/*.cpp")

# set the output directory
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# add the executable
add_executable(Mlcpp ${SOURCES} ${TEST_SOURCES})