add_executable(reference-collapsing main.cc)
target_link_libraries(reference-collapsing GTest::gtest_main)
gtest_discover_tests(reference-collapsing)
