add_executable(map main.cc)
target_link_libraries(map GTest::gtest_main)
gtest_discover_tests(map)
