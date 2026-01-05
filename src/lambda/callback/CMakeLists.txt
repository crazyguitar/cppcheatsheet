add_executable(callback main.cc)
target_link_libraries(callback GTest::gtest_main)
gtest_discover_tests(callback)
