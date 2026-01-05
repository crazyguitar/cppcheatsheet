add_executable(deducing-this main.cc)
target_link_libraries(deducing-this GTest::gtest_main)
target_compile_features(deducing-this PRIVATE cxx_std_23)
gtest_discover_tests(deducing-this)
