add_executable(case_range_test main.cc)
target_link_libraries(case_range_test GTest::gtest_main)
gtest_discover_tests(case_range_test)
