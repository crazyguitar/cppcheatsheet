add_executable(coroutine_echo-server main.cc)
target_link_libraries(coroutine_echo-server PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.echo-server COMMAND coroutine_echo-server)
