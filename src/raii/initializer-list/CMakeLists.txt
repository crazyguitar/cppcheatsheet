add_executable(initializer-list main.cc)
target_link_libraries(initializer-list GTest::gtest_main)
gtest_discover_tests(initializer-list)
