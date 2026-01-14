add_executable(ipc_fifo_test main.cc)
target_link_libraries(ipc_fifo_test GTest::gtest_main)
gtest_discover_tests(ipc_fifo_test)
