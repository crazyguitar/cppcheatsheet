add_executable(polymorphism-diamond main.cc)
target_link_libraries(polymorphism-diamond GTest::gtest_main)
gtest_discover_tests(polymorphism-diamond)
