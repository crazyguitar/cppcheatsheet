add_executable(shared-ptr main.cc)
target_link_libraries(shared-ptr GTest::gtest_main)
gtest_discover_tests(shared-ptr)
