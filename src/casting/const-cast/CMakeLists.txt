add_executable(const-cast main.cc)
target_link_libraries(const-cast GTest::gtest_main)
gtest_discover_tests(const-cast)
