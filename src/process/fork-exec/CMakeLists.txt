add_executable(fork_exec_test main.cc)
target_link_libraries(fork_exec_test GTest::gtest_main)
gtest_discover_tests(fork_exec_test)
