add_executable(exception-safety main.cc)
target_link_libraries(exception-safety GTest::gtest_main)
gtest_discover_tests(exception-safety)
