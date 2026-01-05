add_executable(iterator-bounds main.cc)
target_link_libraries(iterator-bounds GTest::gtest_main)
gtest_discover_tests(iterator-bounds)
