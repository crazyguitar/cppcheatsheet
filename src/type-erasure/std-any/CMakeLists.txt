add_executable(type-erasure-std-any main.cc)
target_link_libraries(type-erasure-std-any GTest::gtest_main)
gtest_discover_tests(type-erasure-std-any)
