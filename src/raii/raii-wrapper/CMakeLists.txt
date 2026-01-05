add_executable(raii-wrapper main.cc)
target_link_libraries(raii-wrapper GTest::gtest_main)
gtest_discover_tests(raii-wrapper)
