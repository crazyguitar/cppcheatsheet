add_executable(unix_socket_test main.cc)
target_link_libraries(unix_socket_test GTest::gtest_main)
gtest_discover_tests(unix_socket_test)
