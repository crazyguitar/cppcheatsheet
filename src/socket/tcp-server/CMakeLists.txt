add_executable(tcp_server_test main.cc)
target_link_libraries(tcp_server_test GTest::gtest_main)
gtest_discover_tests(tcp_server_test)
