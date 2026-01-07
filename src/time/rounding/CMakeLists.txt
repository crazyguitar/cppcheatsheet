add_executable(rounding main.cc)
target_link_libraries(rounding GTest::gtest_main)
target_compile_features(rounding PRIVATE cxx_std_20)
gtest_discover_tests(rounding)
