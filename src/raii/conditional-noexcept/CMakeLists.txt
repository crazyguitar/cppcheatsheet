add_executable(conditional-noexcept main.cc)
target_link_libraries(conditional-noexcept GTest::gtest_main)
gtest_discover_tests(conditional-noexcept)
