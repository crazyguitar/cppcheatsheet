add_executable(nested_func_test main.cc)
target_link_libraries(nested_func_test GTest::gtest_main)
gtest_discover_tests(nested_func_test)
