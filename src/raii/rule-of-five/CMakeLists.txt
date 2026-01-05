add_executable(rule-of-five main.cc)
target_link_libraries(rule-of-five GTest::gtest_main)
gtest_discover_tests(rule-of-five)
