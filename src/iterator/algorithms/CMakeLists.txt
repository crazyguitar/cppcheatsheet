add_executable(iterator-algorithms main.cc)
target_link_libraries(iterator-algorithms GTest::gtest_main)
gtest_discover_tests(iterator-algorithms)
