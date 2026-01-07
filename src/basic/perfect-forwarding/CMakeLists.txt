add_executable(perfect-forwarding main.cc)
target_link_libraries(perfect-forwarding GTest::gtest_main)
gtest_discover_tests(perfect-forwarding)
