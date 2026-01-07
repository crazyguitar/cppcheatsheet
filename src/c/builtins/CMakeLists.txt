add_executable(builtins_test main.cc)
target_link_libraries(builtins_test GTest::gtest_main)
gtest_discover_tests(builtins_test)
