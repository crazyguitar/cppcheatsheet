add_executable(mutable main.cc)
target_link_libraries(mutable GTest::gtest_main)
gtest_discover_tests(mutable)
