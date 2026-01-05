add_executable(constexpr-variable main.cc)
target_link_libraries(constexpr-variable GTest::gtest_main)
gtest_discover_tests(constexpr-variable)
