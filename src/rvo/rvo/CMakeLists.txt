add_executable(rvo main.cc)
target_link_libraries(rvo GTest::gtest_main)
gtest_discover_tests(rvo)
