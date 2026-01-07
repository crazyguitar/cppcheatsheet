add_executable(attributes_test main.cc)
target_link_libraries(attributes_test GTest::gtest_main)
gtest_discover_tests(attributes_test)
