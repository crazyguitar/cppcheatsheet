add_executable(concat-perf main.cc)
target_link_libraries(concat-perf GTest::gtest_main)
gtest_discover_tests(concat-perf)
