add_executable(iterator-iota main.cc)
target_link_libraries(iterator-iota GTest::gtest_main)
gtest_discover_tests(iterator-iota)
