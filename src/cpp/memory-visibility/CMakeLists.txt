add_executable(memory-visibility_test main.cc)
target_link_libraries(memory-visibility_test GTest::gtest_main)
gtest_discover_tests(memory-visibility_test)
