add_executable(future-async main.cc)
target_link_libraries(future-async GTest::gtest_main)
target_compile_features(future-async PRIVATE cxx_std_17)
gtest_discover_tests(future-async)
