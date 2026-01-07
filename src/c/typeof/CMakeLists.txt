add_executable(typeof_test main.cc)
target_link_libraries(typeof_test GTest::gtest_main)
gtest_discover_tests(typeof_test)
