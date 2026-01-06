add_executable(coroutine_result main.cc)
target_link_libraries(coroutine_result PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.result COMMAND coroutine_result)
