add_executable(poll_server_test main.cc)
target_link_libraries(poll_server_test GTest::gtest_main)
gtest_discover_tests(poll_server_test)
