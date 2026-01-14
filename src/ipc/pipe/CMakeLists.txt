add_executable(ipc_pipe_test main.cc)
target_link_libraries(ipc_pipe_test GTest::gtest_main)
gtest_discover_tests(ipc_pipe_test)
