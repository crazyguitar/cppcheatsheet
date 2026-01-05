add_executable(timestamp main.cc)
target_link_libraries(timestamp GTest::gtest_main)
gtest_discover_tests(timestamp)
