add_executable(default-constructible main.cc)
target_link_libraries(default-constructible GTest::gtest_main)
target_compile_features(default-constructible PRIVATE cxx_std_20)
gtest_discover_tests(default-constructible)
