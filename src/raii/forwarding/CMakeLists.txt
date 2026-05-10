add_executable(forwarding main.cc)
target_link_libraries(forwarding GTest::gtest_main)
gtest_discover_tests(forwarding)
