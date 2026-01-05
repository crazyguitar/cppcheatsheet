add_executable(conditional main.cc)
target_link_libraries(conditional GTest::gtest_main)
gtest_discover_tests(conditional)
