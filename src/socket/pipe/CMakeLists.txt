add_executable(pipe_test main.cc)
target_link_libraries(pipe_test GTest::gtest_main)
gtest_discover_tests(pipe_test)
