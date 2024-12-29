find_package(GTest REQUIRED)

enable_testing()

if(NOT GTest_FOUND)
  message(FATAL_ERROR "GTest not found")
endif()

file(GLOB NVOCDR_TEST_SRC *.cpp)

add_executable(nvocdr_test ${NVOCDR_TEST_SRC})
target_link_libraries(nvocdr_test PRIVATE ${OpenCV_LIBS} GTest::gtest_main glog::glog nvocdr )

# add_test(NAME ThisIsTestA COMMAND nvocdr_basic_test WORKING_DIRECTORY ${DIRECTORY_WITH_TEST_DATA})

# include(GoogleTest)
# gtest_discover_tests(nvocdr_basic_test)