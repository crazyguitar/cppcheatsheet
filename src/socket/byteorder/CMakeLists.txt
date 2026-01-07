add_executable(byteorder_test main.cc)
target_link_libraries(byteorder_test GTest::gtest_main)
gtest_discover_tests(byteorder_test)
