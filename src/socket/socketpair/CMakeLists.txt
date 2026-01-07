add_executable(socketpair_test main.cc)
target_link_libraries(socketpair_test GTest::gtest_main)
gtest_discover_tests(socketpair_test)
