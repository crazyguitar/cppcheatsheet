add_executable(print main.cc)
target_compile_features(print PRIVATE cxx_std_23)
target_link_libraries(print GTest::gtest_main)
gtest_discover_tests(print)
