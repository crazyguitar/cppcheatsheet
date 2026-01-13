add_executable(require-basic main.cc)
target_link_libraries(require-basic GTest::gtest_main)
target_compile_features(require-basic PRIVATE cxx_std_20)
gtest_discover_tests(require-basic)
