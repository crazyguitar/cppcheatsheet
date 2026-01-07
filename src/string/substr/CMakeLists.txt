add_executable(substr main.cc)
target_link_libraries(substr GTest::gtest_main)
gtest_discover_tests(substr)
