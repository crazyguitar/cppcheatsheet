add_executable(bit-manipulation main.cc)
target_link_libraries(bit-manipulation GTest::gtest_main)
gtest_discover_tests(bit-manipulation)
