add_executable(uniform-initialization main.cc)
target_link_libraries(uniform-initialization GTest::gtest_main)
gtest_discover_tests(uniform-initialization)
