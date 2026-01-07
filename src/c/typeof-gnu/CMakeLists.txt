add_executable(typeof_gnu_test main.cc)
target_link_libraries(typeof_gnu_test GTest::gtest_main)
gtest_discover_tests(typeof_gnu_test)
