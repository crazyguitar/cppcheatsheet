add_executable(read_lines_test main.cc)
target_link_libraries(read_lines_test GTest::gtest_main)
gtest_discover_tests(read_lines_test)
