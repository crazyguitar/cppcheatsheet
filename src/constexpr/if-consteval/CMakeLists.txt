add_executable(if-consteval main.cc)
target_link_libraries(if-consteval GTest::gtest_main)
target_compile_features(if-consteval PRIVATE cxx_std_23)
gtest_discover_tests(if-consteval)
