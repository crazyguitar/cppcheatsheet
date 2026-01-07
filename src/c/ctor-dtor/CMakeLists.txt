add_executable(ctor_dtor_test main.cc)
target_link_libraries(ctor_dtor_test GTest::gtest_main)
gtest_discover_tests(ctor_dtor_test)
