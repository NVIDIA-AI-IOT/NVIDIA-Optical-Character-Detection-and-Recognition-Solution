find_package(GTest REQUIRED)

enable_testing()

if(NOT GTest_FOUND)
  message(FATAL_ERROR "GTest not found")
endif()

file(GLOB NVOCDR_TEST_SRC test_model*.cpp)

add_executable(test_model ${NVOCDR_TEST_SRC})
target_link_libraries(test_model PRIVATE ${OpenCV_LIBS} GTest::gtest_main glog::glog nvocdr)

add_executable(test_tokenizer test_tokenizer_bpe.cpp ${CMAKE_SOURCE_DIR}/src/impl/tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE GTest::gtest_main glog::glog)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/src/impl)

# add_test(NAME ThisIsTestA COMMAND nvocdr_basic_test WORKING_DIRECTORY ${DIRECTORY_WITH_TEST_DATA})

# include(GoogleTest)
# gtest_discover_tests(nvocdr_basic_test)