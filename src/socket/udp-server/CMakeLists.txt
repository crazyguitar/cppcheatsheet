add_executable(udp_server_test main.cc)
target_link_libraries(udp_server_test GTest::gtest_main)
gtest_discover_tests(udp_server_test)
