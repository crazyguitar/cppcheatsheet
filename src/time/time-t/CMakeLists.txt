add_executable(time-t main.cc)
target_link_libraries(time-t GTest::gtest_main)
gtest_discover_tests(time-t)
