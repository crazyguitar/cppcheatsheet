add_executable(coroutine_coro main.cc)
target_link_libraries(coroutine_coro PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.coro COMMAND coroutine_coro)
