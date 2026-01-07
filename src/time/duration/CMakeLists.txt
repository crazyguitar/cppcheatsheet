add_executable(duration main.cc)
target_link_libraries(duration GTest::gtest_main)
gtest_discover_tests(duration)
