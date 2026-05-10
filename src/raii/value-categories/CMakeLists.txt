add_executable(value-categories main.cc)
target_link_libraries(value-categories GTest::gtest_main)
gtest_discover_tests(value-categories)
