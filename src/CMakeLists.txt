# find_package(OpenMP)

include(FetchContent)

FetchContent_Declare(json 
                     GIT_REPOSITORY  https://github.com/nlohmann/json.git
                     GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)
file(GLOB NVOCDR_SRC nvocdr.cpp impl/*.cpp impl/*.cu)

add_library(nvocdr SHARED ${NVOCDR_SRC})

target_include_directories(nvocdr PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvocdr PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TRT_PLUGIN_LIBS} glog::glog nlohmann_json::nlohmann_json)

# if (OPENMP_FOUND)
#     message("-- OpenMP found")
#     target_link_libraries(nvocdr PUBLIC OpenMP::OpenMP_CXX)
# endif()

# todo(shuohanc) add more supported arch
set_target_properties(nvocdr PROPERTIES CUDA_ARCHITECTURES "50;72;86")
target_compile_features(nvocdr PUBLIC cuda_std_11)

# set_target_properties(nvocdr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# todo(shuohanc) add install commands

# install(
#   TARGETS
#    nvocdr
#   LIBRARY DESTINATION backends/nvocdr
#   RUNTIME DESTINATION backends/nvocdr
# )