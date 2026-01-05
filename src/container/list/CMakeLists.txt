add_executable(list main.cc)
target_link_libraries(list GTest::gtest_main)
gtest_discover_tests(list)
