add_executable(init-capture main.cc)
target_link_libraries(init-capture GTest::gtest_main)
gtest_discover_tests(init-capture)
