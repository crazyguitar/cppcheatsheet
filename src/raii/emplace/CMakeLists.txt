add_executable(emplace main.cc)
target_link_libraries(emplace GTest::gtest_main)
gtest_discover_tests(emplace)
