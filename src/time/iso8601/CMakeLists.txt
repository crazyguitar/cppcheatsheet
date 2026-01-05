add_executable(iso8601 main.cc)
target_link_libraries(iso8601 GTest::gtest_main)
gtest_discover_tests(iso8601)
