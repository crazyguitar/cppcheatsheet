add_executable(daemon_test main.cc)
target_link_libraries(daemon_test GTest::gtest_main)
gtest_discover_tests(daemon_test)
