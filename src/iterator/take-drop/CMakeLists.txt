add_executable(iterator-take-drop main.cc)
target_link_libraries(iterator-take-drop GTest::gtest_main)
gtest_discover_tests(iterator-take-drop)
