add_executable(string-view main.cc)
target_link_libraries(string-view GTest::gtest_main)
gtest_discover_tests(string-view)
