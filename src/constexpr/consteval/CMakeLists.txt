add_executable(consteval main.cc)
target_link_libraries(consteval GTest::gtest_main)
target_compile_features(consteval PRIVATE cxx_std_20)
gtest_discover_tests(consteval)
