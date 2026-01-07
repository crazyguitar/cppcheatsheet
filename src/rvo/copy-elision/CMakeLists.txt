add_executable(copy-elision main.cc)
target_link_libraries(copy-elision GTest::gtest_main)
gtest_discover_tests(copy-elision)
