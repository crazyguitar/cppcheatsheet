add_executable(fstat_size_test main.cc)
target_link_libraries(fstat_size_test GTest::gtest_main)
gtest_discover_tests(fstat_size_test)
