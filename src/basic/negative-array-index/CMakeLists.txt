add_executable(negative-array-index main.cc)
target_link_libraries(negative-array-index GTest::gtest_main)
gtest_discover_tests(negative-array-index)
