add_executable(fifo_test main.cc)
target_link_libraries(fifo_test GTest::gtest_main)
gtest_discover_tests(fifo_test)
