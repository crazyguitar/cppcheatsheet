add_executable(pthread_server_test main.cc)
target_link_libraries(pthread_server_test GTest::gtest_main Threads::Threads)
gtest_discover_tests(pthread_server_test)
