add_executable(getaddrinfo_test main.cc)
target_link_libraries(getaddrinfo_test GTest::gtest_main)
gtest_discover_tests(getaddrinfo_test)
