add_executable(polymorphism-multiple-inheritance main.cc)
target_link_libraries(polymorphism-multiple-inheritance GTest::gtest_main)
gtest_discover_tests(polymorphism-multiple-inheritance)
