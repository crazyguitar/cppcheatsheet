add_executable(type-erasure-std-function main.cc)
target_link_libraries(type-erasure-std-function GTest::gtest_main)
gtest_discover_tests(type-erasure-std-function)
