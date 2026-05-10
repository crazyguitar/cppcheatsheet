add_executable(polymorphism-virtual-destructor main.cc)
target_link_libraries(polymorphism-virtual-destructor GTest::gtest_main)
gtest_discover_tests(polymorphism-virtual-destructor)
