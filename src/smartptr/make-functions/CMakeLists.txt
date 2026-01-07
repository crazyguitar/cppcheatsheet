add_executable(make-functions main.cc)
target_link_libraries(make-functions GTest::gtest_main)
gtest_discover_tests(make-functions)
