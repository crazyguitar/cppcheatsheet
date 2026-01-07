add_executable(copy_file_test main.cc)
target_link_libraries(copy_file_test GTest::gtest_main)
gtest_discover_tests(copy_file_test)
