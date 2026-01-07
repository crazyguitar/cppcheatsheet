add_executable(set main.cc)
target_link_libraries(set GTest::gtest_main)
gtest_discover_tests(set)
