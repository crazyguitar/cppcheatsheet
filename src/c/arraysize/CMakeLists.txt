add_executable(arraysize_test main.cc)
target_link_libraries(arraysize_test GTest::gtest_main)
gtest_discover_tests(arraysize_test)
