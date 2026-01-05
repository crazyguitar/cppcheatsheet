add_executable(unevaluated main.cc)
target_link_libraries(unevaluated GTest::gtest_main)
target_compile_features(unevaluated PRIVATE cxx_std_20)
gtest_discover_tests(unevaluated)
