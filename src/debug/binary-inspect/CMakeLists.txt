add_executable(binary-inspect_test main.cc helper.cc)
target_link_libraries(binary-inspect_test GTest::gtest_main)
gtest_discover_tests(binary-inspect_test)

add_test(
  NAME Debug.BinaryInspect.BashTools
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/inspect.sh $<TARGET_FILE:binary-inspect_test>
)
