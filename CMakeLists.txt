project(NV_OCDR LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.24)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


if(OpenCV_FOUND)
    message("Found OpenCV, includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

if(CUDA_FOUND)
    message("Found CUDA, includes" ${CUDA_INCLUDE_DIRS})
endif()

add_subdirectory(src)
add_subdirectory(c++_samples)