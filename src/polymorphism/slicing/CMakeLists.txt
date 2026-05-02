add_executable(polymorphism-slicing main.cc)
target_link_libraries(polymorphism-slicing GTest::gtest_main)
gtest_discover_tests(polymorphism-slicing)
