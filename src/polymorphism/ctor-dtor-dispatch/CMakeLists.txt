add_executable(polymorphism-ctor-dtor-dispatch main.cc)
target_link_libraries(polymorphism-ctor-dtor-dispatch GTest::gtest_main)
gtest_discover_tests(polymorphism-ctor-dtor-dispatch)
