add_executable(coroutine_generator main.cc)
target_link_libraries(coroutine_generator PRIVATE coro_io GTest::gtest_main)
add_test(NAME Coroutine.generator COMMAND coroutine_generator)
