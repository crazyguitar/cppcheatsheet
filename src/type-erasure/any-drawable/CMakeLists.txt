add_executable(type-erasure-any-drawable main.cc)
target_link_libraries(type-erasure-any-drawable GTest::gtest_main)
gtest_discover_tests(type-erasure-any-drawable)
