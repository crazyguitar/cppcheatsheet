add_executable(anon_struct_test main.cc)
target_link_libraries(anon_struct_test GTest::gtest_main)
gtest_discover_tests(anon_struct_test)
