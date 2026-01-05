add_executable(captureless main.cc)
target_link_libraries(captureless GTest::gtest_main)
gtest_discover_tests(captureless)
