add_executable(template-lambda main.cc)
target_link_libraries(template-lambda GTest::gtest_main)
target_compile_features(template-lambda PRIVATE cxx_std_20)
gtest_discover_tests(template-lambda)
