add_executable(rwlock_test main.cc)
target_link_libraries(rwlock_test GTest::gtest_main pthread)
gtest_discover_tests(rwlock_test)
