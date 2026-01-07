add_executable(alignas_test main.cc)
target_link_libraries(alignas_test GTest::gtest_main)
gtest_discover_tests(alignas_test)
