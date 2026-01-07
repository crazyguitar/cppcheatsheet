add_executable(static-cast main.cc)
target_link_libraries(static-cast GTest::gtest_main)
gtest_discover_tests(static-cast)
