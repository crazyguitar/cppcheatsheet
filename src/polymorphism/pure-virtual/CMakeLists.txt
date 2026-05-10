add_executable(polymorphism-pure-virtual main.cc)
target_link_libraries(polymorphism-pure-virtual GTest::gtest_main)
gtest_discover_tests(polymorphism-pure-virtual)
