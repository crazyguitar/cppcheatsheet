add_executable(pack-capture main.cc)
target_link_libraries(pack-capture GTest::gtest_main)
target_compile_features(pack-capture PRIVATE cxx_std_20)
gtest_discover_tests(pack-capture)
