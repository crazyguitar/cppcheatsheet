add_executable(polymorphism-nvi main.cc)
target_link_libraries(polymorphism-nvi GTest::gtest_main)
gtest_discover_tests(polymorphism-nvi)
