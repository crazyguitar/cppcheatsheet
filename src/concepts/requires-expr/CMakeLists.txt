add_executable(concepts-requires-expr main.cc)
target_link_libraries(concepts-requires-expr GTest::gtest_main)
target_compile_features(concepts-requires-expr PRIVATE cxx_std_20)
gtest_discover_tests(concepts-requires-expr)
