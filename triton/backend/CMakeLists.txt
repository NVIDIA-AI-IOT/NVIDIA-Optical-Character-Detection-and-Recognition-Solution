
# https://github.com/triton-inference-server/backend/blob/main/examples/backends/recommended/CMakeLists.txt

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r24.12" CACHE STRING "Tag for triton-inference-server/backend repo")

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(libtriton_nvocdr.ldscript libtriton_nvocdr.ldscript COPYONLY)

add_library(
    triton_nvocdr SHARED
    triton_nvocdr_backend.cpp
)

target_link_libraries(
    triton_nvocdr
  PRIVATE
    nvocdr
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

install(
  TARGETS
    triton_nvocdr
  LIBRARY DESTINATION backends/nvocdr
  RUNTIME DESTINATION backends/nvocdr
)

install(FILES config.pbtxt DESTINATION backends/nvocdr)
install(FILES placeholder DESTINATION backends/nvocdr/1)