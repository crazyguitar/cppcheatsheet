add_executable(signal_mask_test main.cc)
target_link_libraries(signal_mask_test GTest::gtest_main)
gtest_discover_tests(signal_mask_test)
