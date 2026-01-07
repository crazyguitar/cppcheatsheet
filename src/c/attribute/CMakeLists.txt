add_executable(attribute_test main.cc)
target_link_libraries(attribute_test GTest::gtest_main)
gtest_discover_tests(attribute_test)
