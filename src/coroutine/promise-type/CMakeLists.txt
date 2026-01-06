add_executable(coroutine_promise-type main.cc)
target_link_libraries(coroutine_promise-type PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.promise-type COMMAND coroutine_promise-type)
