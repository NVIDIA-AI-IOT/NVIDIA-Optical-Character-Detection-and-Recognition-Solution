project(NV_OCDR LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(glog 0.4.0 REQUIRED)

option(BUILD_TESTS "Build test cases" ON)
option(BUILD_SAMPLES "Build samples" ON)

if(OpenCV_FOUND)
    message("-- Found OpenCV, includes headers: " ${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR  "OpenCV not found!!!")
endif()

if(CUDA_FOUND)
    message("-- Found CUDA, include headers:" ${CUDA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR  "cuda not found!!!")
endif()

if(glog_FOUND)
   message("-- Found GLOG, version: " ${glog_VERSION})
else()
   message(FATAL_ERROR  "glog not found!!!")
endif()

add_subdirectory(src)
add_subdirectory(py_bind)

if(BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()