add_executable(case-conversion main.cc)
target_link_libraries(case-conversion GTest::gtest_main)
gtest_discover_tests(case-conversion)
