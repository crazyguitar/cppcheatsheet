add_executable(iterator-basics main.cc)
target_link_libraries(iterator-basics GTest::gtest_main)
gtest_discover_tests(iterator-basics)
