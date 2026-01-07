add_executable(iterator-compose main.cc)
target_link_libraries(iterator-compose GTest::gtest_main)
gtest_discover_tests(iterator-compose)
