add_executable(two-phase-init main.cc)
target_link_libraries(two-phase-init GTest::gtest_main)
gtest_discover_tests(two-phase-init)
