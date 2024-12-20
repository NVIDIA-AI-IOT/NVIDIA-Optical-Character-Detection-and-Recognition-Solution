list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    if(${libName}_lib)
        message("TensorRT found: " ${libName})
    endif()
    list(APPEND TRT_PLUGIN_LIBS "${${libName}_lib}")
endforeach()


file(GLOB NVOCDR_SRC *.cpp *.cu)
file(GLOB NVOCDR_HDR *.h)

add_library(nvocdr SHARED ${NVOCDR_SRC})

target_include_directories(nvocdr PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nvocdr PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TRT_PLUGIN_LIBS})

set_target_properties(nvocdr PROPERTIES CUDA_ARCHITECTURES "50;72;86")

target_compile_features(nvocdr PUBLIC cxx_std_11)
target_compile_features(nvocdr PUBLIC cuda_std_11)

# set_target_properties(nvocdr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# todo(shuohanc) add install commands

