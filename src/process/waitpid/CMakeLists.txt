add_executable(waitpid_test main.cc)
target_link_libraries(waitpid_test GTest::gtest_main)
gtest_discover_tests(waitpid_test)
