add_executable(coroutine_future main.cc)
target_link_libraries(coroutine_future PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.future COMMAND coroutine_future)
