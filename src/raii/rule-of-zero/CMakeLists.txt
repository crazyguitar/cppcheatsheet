add_executable(rule-of-zero main.cc)
target_link_libraries(rule-of-zero GTest::gtest_main)
gtest_discover_tests(rule-of-zero)
