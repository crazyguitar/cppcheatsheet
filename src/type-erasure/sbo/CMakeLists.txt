add_executable(type-erasure-sbo main.cc)
target_link_libraries(type-erasure-sbo GTest::gtest_main)
gtest_discover_tests(type-erasure-sbo)
