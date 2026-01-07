add_executable(pid_test main.cc)
target_link_libraries(pid_test GTest::gtest_main)
gtest_discover_tests(pid_test)
