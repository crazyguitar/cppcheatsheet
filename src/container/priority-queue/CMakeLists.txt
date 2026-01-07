add_executable(priority-queue main.cc)
target_link_libraries(priority-queue GTest::gtest_main)
gtest_discover_tests(priority-queue)
