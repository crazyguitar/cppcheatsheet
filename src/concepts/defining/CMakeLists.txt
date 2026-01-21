add_executable(concepts-defining main.cc)
target_link_libraries(concepts-defining GTest::gtest_main)
target_compile_features(concepts-defining PRIVATE cxx_std_20)
gtest_discover_tests(concepts-defining)
