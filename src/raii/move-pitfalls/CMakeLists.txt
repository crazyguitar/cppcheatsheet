add_executable(move-pitfalls main.cc)
target_link_libraries(move-pitfalls GTest::gtest_main)
gtest_discover_tests(move-pitfalls)
