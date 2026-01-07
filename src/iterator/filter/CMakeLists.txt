add_executable(iterator-filter main.cc)
target_link_libraries(iterator-filter GTest::gtest_main)
gtest_discover_tests(iterator-filter)
