add_executable(concepts-subsumption main.cc)
target_link_libraries(concepts-subsumption GTest::gtest_main)
target_compile_features(concepts-subsumption PRIVATE cxx_std_20)
gtest_discover_tests(concepts-subsumption)
