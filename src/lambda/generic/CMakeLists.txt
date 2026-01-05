add_executable(generic main.cc)
target_link_libraries(generic GTest::gtest_main)
gtest_discover_tests(generic)
