add_executable(polymorphic main.cc)
target_link_libraries(polymorphic GTest::gtest_main)
gtest_discover_tests(polymorphic)
