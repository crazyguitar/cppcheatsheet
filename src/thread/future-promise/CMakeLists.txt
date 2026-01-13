add_executable(future-promise main.cc)
target_link_libraries(future-promise GTest::gtest_main)
target_compile_features(future-promise PRIVATE cxx_std_17)
gtest_discover_tests(future-promise)
