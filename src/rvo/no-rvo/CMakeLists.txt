add_executable(no-rvo main.cc)
target_link_libraries(no-rvo GTest::gtest_main)
gtest_discover_tests(no-rvo)
