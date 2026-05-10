add_executable(moved-from-state main.cc)
target_link_libraries(moved-from-state GTest::gtest_main)
gtest_discover_tests(moved-from-state)
