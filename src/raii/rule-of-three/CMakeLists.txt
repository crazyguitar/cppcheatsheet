add_executable(rule-of-three main.cc)
target_link_libraries(rule-of-three GTest::gtest_main)
gtest_discover_tests(rule-of-three)
