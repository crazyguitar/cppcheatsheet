add_executable(fork_wait_test main.cc)
target_link_libraries(fork_wait_test GTest::gtest_main)
gtest_discover_tests(fork_wait_test)
