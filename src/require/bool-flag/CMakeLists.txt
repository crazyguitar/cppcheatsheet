add_executable(require-bool-flag main.cc)
target_link_libraries(require-bool-flag GTest::gtest_main)
target_compile_features(require-bool-flag PRIVATE cxx_std_20)
gtest_discover_tests(require-bool-flag)
