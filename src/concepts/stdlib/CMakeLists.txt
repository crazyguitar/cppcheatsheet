add_executable(concepts-stdlib main.cc)
target_link_libraries(concepts-stdlib GTest::gtest_main)
target_compile_features(concepts-stdlib PRIVATE cxx_std_20)
gtest_discover_tests(concepts-stdlib)
