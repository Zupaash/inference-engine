cmake_minimum_required(VERSION 3.10)
project(NeuralInferenceEngine)

# Use standard modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find all C++ source files in the src/ directory
file(GLOB SOURCES "src/*.cpp")

# Define your executable output
add_executable(inference_engine ${SOURCES})
