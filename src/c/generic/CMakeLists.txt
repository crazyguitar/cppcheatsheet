add_executable(generic_test main.cc)
target_link_libraries(generic_test GTest::gtest_main)
gtest_discover_tests(generic_test)
