add_executable(iterator-transform main.cc)
target_link_libraries(iterator-transform GTest::gtest_main)
gtest_discover_tests(iterator-transform)
