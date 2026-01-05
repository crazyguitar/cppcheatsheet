add_executable(constinit-test main.cc)
target_link_libraries(constinit-test GTest::gtest_main)
target_compile_features(constinit-test PRIVATE cxx_std_20)
gtest_discover_tests(constinit-test)
