add_executable(move-exchange main.cc)
target_link_libraries(move-exchange GTest::gtest_main)
gtest_discover_tests(move-exchange)
