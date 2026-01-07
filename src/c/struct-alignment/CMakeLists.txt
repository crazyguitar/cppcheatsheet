add_executable(struct-alignment_test main.cc)
target_link_libraries(struct-alignment_test GTest::gtest_main)
gtest_discover_tests(struct-alignment_test)
