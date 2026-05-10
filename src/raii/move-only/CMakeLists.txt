add_executable(move-only main.cc)
target_link_libraries(move-only GTest::gtest_main)
gtest_discover_tests(move-only)
