add_executable(no-move-return main.cc)
target_link_libraries(no-move-return GTest::gtest_main)
gtest_discover_tests(no-move-return)
