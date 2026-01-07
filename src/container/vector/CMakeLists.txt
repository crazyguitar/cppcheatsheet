add_executable(vector main.cc)
target_link_libraries(vector GTest::gtest_main)
gtest_discover_tests(vector)
