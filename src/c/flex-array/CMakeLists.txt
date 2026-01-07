add_executable(flex_array_test main.cc)
target_link_libraries(flex_array_test GTest::gtest_main)
gtest_discover_tests(flex_array_test)
