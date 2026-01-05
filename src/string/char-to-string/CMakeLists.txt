add_executable(char-to-string main.cc)
target_link_libraries(char-to-string GTest::gtest_main)
gtest_discover_tests(char-to-string)
