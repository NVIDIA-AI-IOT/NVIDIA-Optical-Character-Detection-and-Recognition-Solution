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


# impl/nvOCDR.cpp 
file(GLOB NVOCDR_SRC nvocdr.cpp impl/MemManager.cpp impl/TRTEngine.cpp impl/OCRNetEngine.cpp impl/OCDNetEngine.cpp impl/nvOCDR.cpp )
# file(GLOB NVOCDR_SRC impl/*.cpp impl/*.cu )

add_library(nvocdr SHARED ${NVOCDR_SRC})

target_include_directories(nvocdr PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvocdr PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TRT_PLUGIN_LIBS} glog::glog)

# todo(shuohanc) add more supported arch
set_target_properties(nvocdr PROPERTIES CUDA_ARCHITECTURES "50;72;86")

target_compile_features(nvocdr PUBLIC cxx_std_11)
target_compile_features(nvocdr PUBLIC cuda_std_11)

# set_target_properties(nvocdr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# todo(shuohanc) add install commands

