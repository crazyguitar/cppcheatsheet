add_executable(require-conjunction main.cc)
target_link_libraries(require-conjunction GTest::gtest_main)
target_compile_features(require-conjunction PRIVATE cxx_std_20)
gtest_discover_tests(require-conjunction)
