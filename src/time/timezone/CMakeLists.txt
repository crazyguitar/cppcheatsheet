add_executable(timezone main.cc)
target_link_libraries(timezone GTest::gtest_main)
gtest_discover_tests(timezone)
