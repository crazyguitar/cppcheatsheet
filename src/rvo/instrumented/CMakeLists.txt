add_executable(instrumented main.cc)
target_link_libraries(instrumented GTest::gtest_main)
gtest_discover_tests(instrumented)
