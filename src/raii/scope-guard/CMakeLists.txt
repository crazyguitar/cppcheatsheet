add_executable(scope-guard main.cc)
target_link_libraries(scope-guard GTest::gtest_main)
gtest_discover_tests(scope-guard)
