add_executable(callable main.cc)
target_link_libraries(callable GTest::gtest_main)
gtest_discover_tests(callable)
