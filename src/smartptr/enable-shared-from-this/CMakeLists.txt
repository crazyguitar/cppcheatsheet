add_executable(enable-shared-from-this main.cc)
target_link_libraries(enable-shared-from-this GTest::gtest_main)
gtest_discover_tests(enable-shared-from-this)
