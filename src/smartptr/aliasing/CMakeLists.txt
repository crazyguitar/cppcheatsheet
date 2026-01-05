add_executable(aliasing main.cc)
target_link_libraries(aliasing GTest::gtest_main)
gtest_discover_tests(aliasing)
