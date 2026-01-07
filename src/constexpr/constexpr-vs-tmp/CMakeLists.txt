add_executable(constexpr-vs-tmp main.cc)
target_link_libraries(constexpr-vs-tmp GTest::gtest_main)
gtest_discover_tests(constexpr-vs-tmp)
