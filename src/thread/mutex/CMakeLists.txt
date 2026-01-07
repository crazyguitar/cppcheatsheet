add_executable(mutex_test main.cc)
target_link_libraries(mutex_test GTest::gtest_main pthread)
gtest_discover_tests(mutex_test)
