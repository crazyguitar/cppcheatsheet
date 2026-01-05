add_executable(profiling main.cc)
target_link_libraries(profiling GTest::gtest_main)
gtest_discover_tests(profiling)
