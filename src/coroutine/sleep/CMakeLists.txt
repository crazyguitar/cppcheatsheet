add_executable(coroutine_sleep main.cc)
target_link_libraries(coroutine_sleep PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.sleep COMMAND coroutine_sleep)
