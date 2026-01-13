add_executable(future-basic main.cc)
target_link_libraries(future-basic GTest::gtest_main)
target_compile_features(future-basic PRIVATE cxx_std_17)
gtest_discover_tests(future-basic)
