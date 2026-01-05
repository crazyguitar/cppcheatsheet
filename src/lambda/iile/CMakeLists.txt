add_executable(iile main.cc)
target_link_libraries(iile GTest::gtest_main)
gtest_discover_tests(iile)
