add_executable(polymorphism-virtual-functions main.cc)
target_link_libraries(polymorphism-virtual-functions GTest::gtest_main)
gtest_discover_tests(polymorphism-virtual-functions)
