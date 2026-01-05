add_executable(constexpr-lambda main.cc)
target_link_libraries(constexpr-lambda GTest::gtest_main)
gtest_discover_tests(constexpr-lambda)
