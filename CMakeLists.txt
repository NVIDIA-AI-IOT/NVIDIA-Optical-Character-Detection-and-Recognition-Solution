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
option(BUILD_PYBIND "Build python binding" ON)
option(BUILD_TRITON "Build triton" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# find tensorrt 
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
# list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    if(${libName}_lib)
        message("-- TensorRT lib found: " ${libName})
    endif()
    list(APPEND TRT_PLUGIN_LIBS "${${libName}_lib}")
endforeach()


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

if(BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

if(BUILD_TESTS)
  add_subdirectory(pybind)
endif()

if(BUILD_TESTS)
  add_subdirectory(triton/backend)
endif()