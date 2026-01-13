add_executable(require-disjunction main.cc)
target_link_libraries(require-disjunction GTest::gtest_main)
target_compile_features(require-disjunction PRIVATE cxx_std_20)
gtest_discover_tests(require-disjunction)
