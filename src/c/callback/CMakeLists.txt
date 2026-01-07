add_executable(callback_test main.cc)
target_link_libraries(callback_test GTest::gtest_main)
gtest_discover_tests(callback_test)
