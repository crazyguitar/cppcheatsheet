add_executable(lseek_size_test main.cc)
target_link_libraries(lseek_size_test GTest::gtest_main)
gtest_discover_tests(lseek_size_test)
