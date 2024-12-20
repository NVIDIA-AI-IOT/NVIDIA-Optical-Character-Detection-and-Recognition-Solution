# add sampel name here, w/o .cpp suffix
list(APPEND SAMPLE_NAMES "simple_inference")
list(APPEND SAMPLE_NAMES "simple_inference_vit")
list(APPEND SAMPLE_NAMES "simple_inference_ClipOCR")

list(APPEND SAMPLE_NAMES "crop_based_inference")
list(APPEND SAMPLE_NAMES "crop_based_inference_vit")

foreach(sample_name ${SAMPLE_NAMES})
    add_executable(${sample_name} ${sample_name}.cpp)
    target_include_directories(${sample_name} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${sample_name} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvocdr)
endforeach()




