# add sampel name here, w/o .cpp suffix
list(APPEND SAMPLE_NAMES "sample_inference")
# list(APPEND SAMPLE_NAMES "simple_inference_vit")
# list(APPEND SAMPLE_NAMES "simple_inference_ClipOCR")

# list(APPEND SAMPLE_NAMES "crop_based_inference")
# list(APPEND SAMPLE_NAMES "crop_based_inference_vit")

find_package(Boost COMPONENTS program_options REQUIRED)

foreach(sample_name ${SAMPLE_NAMES})
    add_executable(${sample_name} ${sample_name}.cpp base_sample.hpp)
    target_include_directories(${sample_name} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${sample_name} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvocdr)
endforeach()

# add_library(sample_lib SHARED base_sample.hpp)
# target_include_directories(sample_lib PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(sample_lib PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvocdr)
# set_target_properties(sample_lib PROPERTIES LINKER_LANGUAGE CXX)




