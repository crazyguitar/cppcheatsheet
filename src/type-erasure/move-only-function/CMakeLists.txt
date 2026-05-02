add_executable(type-erasure-move-only-function main.cc)
target_link_libraries(type-erasure-move-only-function GTest::gtest_main)
gtest_discover_tests(type-erasure-move-only-function)
