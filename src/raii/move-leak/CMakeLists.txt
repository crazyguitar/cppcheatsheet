add_executable(move-leak main.cc)
target_link_libraries(move-leak GTest::gtest_main)
gtest_discover_tests(move-leak)
