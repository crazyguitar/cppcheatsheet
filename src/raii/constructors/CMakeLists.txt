add_executable(constructors main.cc)
target_link_libraries(constructors GTest::gtest_main)
gtest_discover_tests(constructors)
