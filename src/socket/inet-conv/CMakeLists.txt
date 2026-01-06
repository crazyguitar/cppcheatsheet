add_executable(inet_conv_test main.cc)
target_link_libraries(inet_conv_test GTest::gtest_main)
gtest_discover_tests(inet_conv_test)
