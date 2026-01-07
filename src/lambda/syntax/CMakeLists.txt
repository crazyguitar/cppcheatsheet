add_executable(syntax main.cc)
target_link_libraries(syntax GTest::gtest_main)
gtest_discover_tests(syntax)
