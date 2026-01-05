add_executable(join main.cc)
target_link_libraries(join GTest::gtest_main)
gtest_discover_tests(join)
