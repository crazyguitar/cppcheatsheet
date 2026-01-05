add_executable(from-timestamp main.cc)
target_link_libraries(from-timestamp GTest::gtest_main)
gtest_discover_tests(from-timestamp)
