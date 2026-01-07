add_executable(popen_cmd_test main.cc)
target_link_libraries(popen_cmd_test GTest::gtest_main)
gtest_discover_tests(popen_cmd_test)
