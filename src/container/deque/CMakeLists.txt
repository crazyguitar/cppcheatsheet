add_executable(deque main.cc)
target_link_libraries(deque GTest::gtest_main)
gtest_discover_tests(deque)
