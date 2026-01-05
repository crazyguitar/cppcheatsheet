add_executable(weak-ptr main.cc)
target_link_libraries(weak-ptr GTest::gtest_main)
gtest_discover_tests(weak-ptr)
