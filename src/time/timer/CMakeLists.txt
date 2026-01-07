add_executable(timer main.cc)
target_link_libraries(timer GTest::gtest_main)
target_compile_features(timer PRIVATE cxx_std_20)
gtest_discover_tests(timer)
