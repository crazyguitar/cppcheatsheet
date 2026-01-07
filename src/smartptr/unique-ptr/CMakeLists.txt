add_executable(unique-ptr main.cc)
target_link_libraries(unique-ptr GTest::gtest_main)
gtest_discover_tests(unique-ptr)
