#include <gtest/gtest.h>

#include <future>
#include <thread>

TEST(FuturePromise, SetValue) {
  std::promise<int> prom;
  std::future<int> fut = prom.get_future();

  std::thread t([&prom] {
    prom.set_value(42);  // fulfill the promise
  });

  EXPECT_EQ(fut.get(), 42);  // blocks until value set
  t.join();
}

TEST(FuturePromise, SetException) {
  std::promise<int> prom;
  std::future<int> fut = prom.get_future();

  std::thread t([&prom] { prom.set_exception(std::make_exception_ptr(std::runtime_error("error"))); });

  EXPECT_THROW(fut.get(), std::runtime_error);
  t.join();
}
