add_executable(atomics_test main.cc)
target_link_libraries(atomics_test GTest::gtest_main)
gtest_discover_tests(atomics_test)
