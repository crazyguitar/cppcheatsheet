add_executable(require-constexpr-like main.cc)
target_link_libraries(require-constexpr-like GTest::gtest_main)
target_compile_features(require-constexpr-like PRIVATE cxx_std_20)
gtest_discover_tests(require-constexpr-like)
