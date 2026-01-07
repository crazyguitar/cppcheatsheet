add_executable(dynamic-cast main.cc)
target_link_libraries(dynamic-cast GTest::gtest_main)
gtest_discover_tests(dynamic-cast)
