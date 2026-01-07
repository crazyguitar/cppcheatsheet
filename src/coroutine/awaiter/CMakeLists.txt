add_executable(coroutine_awaiter main.cc)
target_link_libraries(coroutine_awaiter PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.awaiter COMMAND coroutine_awaiter)
