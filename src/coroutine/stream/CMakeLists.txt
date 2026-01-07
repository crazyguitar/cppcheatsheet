add_executable(coroutine_stream main.cc)
target_link_libraries(coroutine_stream PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.stream COMMAND coroutine_stream)
