add_executable(constexpr-function main.cc)
target_link_libraries(constexpr-function GTest::gtest_main)
gtest_discover_tests(constexpr-function)
