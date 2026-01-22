add_executable(self-move main.cc)
target_link_libraries(self-move GTest::gtest_main)
gtest_discover_tests(self-move)
