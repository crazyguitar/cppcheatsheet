add_executable(literals main.cc)
target_link_libraries(literals GTest::gtest_main)
gtest_discover_tests(literals)
