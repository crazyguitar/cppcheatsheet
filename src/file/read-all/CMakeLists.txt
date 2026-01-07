add_executable(read_all_test main.cc)
target_link_libraries(read_all_test GTest::gtest_main)
gtest_discover_tests(read_all_test)
