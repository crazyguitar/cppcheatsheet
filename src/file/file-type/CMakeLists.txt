add_executable(file_type_test main.cc)
target_link_libraries(file_type_test GTest::gtest_main)
gtest_discover_tests(file_type_test)
