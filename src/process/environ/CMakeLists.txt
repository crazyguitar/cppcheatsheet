add_executable(environ_test main.cc)
target_link_libraries(environ_test GTest::gtest_main)
gtest_discover_tests(environ_test)
