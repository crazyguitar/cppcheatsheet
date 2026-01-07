add_executable(tree_walk_test main.cc)
target_link_libraries(tree_walk_test GTest::gtest_main)
gtest_discover_tests(tree_walk_test)
