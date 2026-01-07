add_executable(c-linkage main.cc)
target_link_libraries(c-linkage GTest::gtest_main)
gtest_discover_tests(c-linkage)
