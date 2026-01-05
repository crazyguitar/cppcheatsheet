add_executable(constexpr-if main.cc)
target_link_libraries(constexpr-if GTest::gtest_main)
target_compile_features(constexpr-if PRIVATE cxx_std_17)
gtest_discover_tests(constexpr-if)
