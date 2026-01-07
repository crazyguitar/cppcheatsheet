add_executable(split main.cc)
target_link_libraries(split GTest::gtest_main)
gtest_discover_tests(split)
