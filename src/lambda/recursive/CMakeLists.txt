add_executable(recursive main.cc)
target_link_libraries(recursive GTest::gtest_main)
gtest_discover_tests(recursive)
