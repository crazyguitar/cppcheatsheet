add_executable(hh-mm-ss main.cc)
target_link_libraries(hh-mm-ss GTest::gtest_main)
target_compile_features(hh-mm-ss PRIVATE cxx_std_20)
gtest_discover_tests(hh-mm-ss)
