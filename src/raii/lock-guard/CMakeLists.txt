add_executable(lock-guard main.cc)
target_link_libraries(lock-guard GTest::gtest_main pthread)
gtest_discover_tests(lock-guard)
