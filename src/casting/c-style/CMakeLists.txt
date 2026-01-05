add_executable(c-style main.cc)
target_link_libraries(c-style GTest::gtest_main)
gtest_discover_tests(c-style)
