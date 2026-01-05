add_executable(reinterpret-cast main.cc)
target_link_libraries(reinterpret-cast GTest::gtest_main)
gtest_discover_tests(reinterpret-cast)
