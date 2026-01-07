add_executable(coroutine_scheduler main.cc)
target_link_libraries(coroutine_scheduler PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.scheduler COMMAND coroutine_scheduler)
