add_executable(coroutine_basics main.cc)
target_link_libraries(coroutine_basics PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.basics COMMAND coroutine_basics)
