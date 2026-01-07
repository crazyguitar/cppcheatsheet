add_executable(sysinfo_test main.cc)
target_link_libraries(sysinfo_test GTest::gtest_main)
gtest_discover_tests(sysinfo_test)
