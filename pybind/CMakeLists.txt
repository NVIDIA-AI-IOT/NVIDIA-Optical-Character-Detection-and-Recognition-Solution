include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.0
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pybind11)


pybind11_add_module(nvocdr_python main.cpp py_nvocdr.hpp)
target_compile_definitions(nvocdr_python PRIVATE VERSION_INFO=${VERSION_INFO})
target_link_libraries(nvocdr_python PRIVATE nvocdr)
target_include_directories(nvocdr_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/impl)

# FILE(COPY test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# todo(shuohanc) add install scripts