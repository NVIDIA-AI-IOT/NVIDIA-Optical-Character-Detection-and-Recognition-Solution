
# https://github.com/triton-inference-server/backend/blob/main/examples/backends/recommended/CMakeLists.txt

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/backend repo")

include(FetchContent)

# FetchContent_Declare(
#   tritoncommon
#   GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
#   GIT_TAG ${TRITON_COMMON_REPO_TAG}
#   GIT_SHALLOW ON
# )
# FetchContent_GetProperties(tritoncommon)
# if(NOT tritoncommon_POPULATED)
#   FetchContent_Populate(tritoncommon)
#   add_subdirectory(${tritoncommon_SOURCE_DIR} ${tritoncommon_BINARY_DIR} EXCLUDE_FROM_ALL )
# endif()

# FetchContent_Declare(
#   tritoncore
#   GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
#   GIT_TAG ${TRITON_CORE_REPO_TAG}
#   GIT_SHALLOW ON
# )
# FetchContent_GetProperties(tritoncore)
# if(NOT tritoncore_POPULATED)
#   FetchContent_Populate(tritoncore)
#   add_subdirectory(${tritoncore_SOURCE_DIR} ${tritoncore_BINARY_DIR} EXCLUDE_FROM_ALL )
# endif()

FetchContent_Declare(
  tritonbackend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_GetProperties(tritonbackend)
if(NOT tritonbackend_POPULATED)
  FetchContent_Populate(tritonbackend)
  add_subdirectory(${tritonbackend_SOURCE_DIR} ${tritonbackend_BINARY_DIR} EXCLUDE_FROM_ALL )
endif()


# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(libtriton_nvocdr.ldscript libtriton_nvocdr.ldscript COPYONLY)

set(IMPL_SRC ${CMAKE_SOURCE_DIR}/src)
file(GLOB NVOCDR_SRC ${IMPL_SRC}/nvocdr.cpp ${IMPL_SRC}/impl/*.cpp ${IMPL_SRC}/impl/*.cu)

add_library(
    triton_nvocdr SHARED
    triton_nvocdr_backend.cpp
    ${NVOCDR_SRC}
)

target_include_directories(triton_nvocdr PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${IMPL_SRC})

set_target_properties(triton_nvocdr PROPERTIES CUDA_ARCHITECTURES "50;72;86")
target_compile_features(triton_nvocdr PUBLIC cuda_std_11)

target_link_libraries(
    triton_nvocdr
  PRIVATE
    ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TRT_PLUGIN_LIBS} glog::glog  nlohmann_json::nlohmann_json
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

set_target_properties(
    triton_nvocdr PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_nvocdr
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_nvocdr.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_nvocdr.ldscript"
  )

install(
  TARGETS
    triton_nvocdr
  LIBRARY DESTINATION backends/nvocdr/1
)

install(FILES config.pbtxt DESTINATION backends/nvocdr)
# install(FILES placeholder DESTINATION backends/nvocdr/1)