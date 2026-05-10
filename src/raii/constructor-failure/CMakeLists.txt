add_executable(constructor-failure main.cc)
target_link_libraries(constructor-failure GTest::gtest_main)
gtest_discover_tests(constructor-failure)
