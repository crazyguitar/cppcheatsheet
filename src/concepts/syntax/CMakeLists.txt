add_executable(concepts-syntax main.cc)
target_link_libraries(concepts-syntax GTest::gtest_main)
target_compile_features(concepts-syntax PRIVATE cxx_std_20)
gtest_discover_tests(concepts-syntax)
