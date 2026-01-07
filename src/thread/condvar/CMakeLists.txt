add_executable(condvar_test main.cc)
target_link_libraries(condvar_test GTest::gtest_main pthread)
gtest_discover_tests(condvar_test)
