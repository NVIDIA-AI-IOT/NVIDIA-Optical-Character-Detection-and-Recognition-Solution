find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(sample_inference sample_inference.cpp base_sample.hpp)
target_include_directories(sample_inference PRIVATE ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sample_inference PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvocdr Boost::program_options)



