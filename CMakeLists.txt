cmake_minimum_required(VERSION 3.6.2)
project(MLcpp)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the path to the header files
include_directories(include)

# specify the path to the source files
file(GLOB SOURCES "src/*.cpp")

# specify the path to the test files
file(GLOB TEST_SOURCES "test/*.cpp")

# add the executable
add_executable(Mlcpp ${SOURCES} ${TEST_SOURCES})