add_executable(polymorphism-override-final main.cc)
target_link_libraries(polymorphism-override-final GTest::gtest_main)
gtest_discover_tests(polymorphism-override-final)
