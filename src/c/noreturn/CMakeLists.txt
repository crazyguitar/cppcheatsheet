add_executable(noreturn_test main.cc)
target_link_libraries(noreturn_test GTest::gtest_main)
gtest_discover_tests(noreturn_test)
