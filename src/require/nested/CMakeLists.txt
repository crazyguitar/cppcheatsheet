add_executable(require-nested main.cc)
target_link_libraries(require-nested GTest::gtest_main)
target_compile_features(require-nested PRIVATE cxx_std_20)
gtest_discover_tests(require-nested)
