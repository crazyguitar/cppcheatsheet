add_executable(constexpr_c_test main.cc)
target_link_libraries(constexpr_c_test GTest::gtest_main)
gtest_discover_tests(constexpr_c_test)
