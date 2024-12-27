find_package(GTest REQUIRED)

enable_testing()

if(NOT GTest_FOUND)
  message(FATAL_ERROR "GTest not found")
endif()

add_executable(nvocdr_basic_test nvocdr_basic_test.cpp)
target_link_libraries(nvocdr_basic_test PRIVATE GTest::gtest_main nvocdr)

# add_test(NAME ThisIsTestA COMMAND nvocdr_basic_test WORKING_DIRECTORY ${DIRECTORY_WITH_TEST_DATA})

# include(GoogleTest)
# gtest_discover_tests(nvocdr_basic_test)