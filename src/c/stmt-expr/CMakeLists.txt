add_executable(stmt_expr_test main.cc)
target_link_libraries(stmt_expr_test GTest::gtest_main)
gtest_discover_tests(stmt_expr_test)
