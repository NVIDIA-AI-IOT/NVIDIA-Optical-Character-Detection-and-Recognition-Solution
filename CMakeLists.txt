project(NV_OCDR LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.24)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(glog REQUIRED)


if(OpenCV_FOUND)
    message("Found OpenCV, includes: " ${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR  "OpenCV not found!!!")
endif()

if(CUDA_FOUND)
    message("Found CUDA, includes" ${CUDA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR  "cuda not found!!!")
endif()

if(glog_FOUND)
   message("Found GLOG")
else()
   message(FATAL_ERROR  "glog not found!!!")
endif()

add_subdirectory(src)
add_subdirectory(c++_samples)