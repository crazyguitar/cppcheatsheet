add_executable(nrvo main.cc)
target_link_libraries(nrvo GTest::gtest_main)
gtest_discover_tests(nrvo)
