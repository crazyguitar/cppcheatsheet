add_executable(addressof main.cc)
target_link_libraries(addressof GTest::gtest_main)
gtest_discover_tests(addressof)
