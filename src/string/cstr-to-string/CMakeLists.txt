add_executable(cstr-to-string main.cc)
target_link_libraries(cstr-to-string GTest::gtest_main)
gtest_discover_tests(cstr-to-string)
