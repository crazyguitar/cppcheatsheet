add_executable(custom-deleter main.cc)
target_link_libraries(custom-deleter GTest::gtest_main)
gtest_discover_tests(custom-deleter)
