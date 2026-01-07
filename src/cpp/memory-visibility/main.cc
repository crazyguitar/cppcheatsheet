// C++ Memory Visibility - std::atomic and memory ordering
//
// This example corresponds to the "C++ Memory Model" section in memory_visibility.rst
//
// Memory Order Hierarchy (weakest to strongest):
//
//   relaxed  ->  acquire/release  ->  seq_cst
//   (atomicity)  (synchronization)    (total order)
//
// Acquire-Release Synchronization:
//
//   Thread A (Producer)          Thread B (Consumer)
//   -------------------          -------------------
//   data = 42;                   while(flag.load(acquire) == 0);
//   flag.store(1, release);  --> sees prior writes
//                                use(data);  // guaranteed 42

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

// 1. Memory Orders

// Relaxed: atomicity only, no ordering
TEST(MemoryOrder, Relaxed) {
  std::atomic<int> counter{0};

  std::thread t1([&] {
    for (int i = 0; i < 1000; ++i) counter.fetch_add(1, std::memory_order_relaxed);
  });

  std::thread t2([&] {
    for (int i = 0; i < 1000; ++i) counter.fetch_add(1, std::memory_order_relaxed);
  });

  t1.join();
  t2.join();
  EXPECT_EQ(counter.load(), 2000);
}

// Acquire-Release: synchronizes producer and consumer
TEST(MemoryOrder, AcquireRelease) {
  std::atomic<int> flag{0};
  int data = 0;

  std::thread producer([&] {
    data = 42;
    flag.store(1, std::memory_order_release);  // Release: prior writes visible
  });

  std::thread consumer([&] {
    while (flag.load(std::memory_order_acquire) == 0);  // Acquire: sees prior writes
    EXPECT_EQ(data, 42);
  });

  producer.join();
  consumer.join();
}

// Sequential Consistency: total ordering (default, strongest)
TEST(MemoryOrder, SeqCst) {
  std::atomic<bool> x{false}, y{false};
  std::atomic<int> z{0};

  std::thread t1([&] { x.store(true); });  // seq_cst default
  std::thread t2([&] { y.store(true); });

  std::thread t3([&] {
    while (!x.load());
    if (y.load()) ++z;
  });

  std::thread t4([&] {
    while (!y.load());
    if (x.load()) ++z;
  });

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  EXPECT_GE(z.load(), 1);  // At least one sees both
}

// 2. Atomic Operations

// Compare-and-swap (CAS)
TEST(AtomicOps, CompareExchange) {
  std::atomic<int> val{0};

  int expected = 0;
  EXPECT_TRUE(val.compare_exchange_strong(expected, 1));
  EXPECT_EQ(val.load(), 1);

  expected = 0;  // Wrong expectation
  EXPECT_FALSE(val.compare_exchange_strong(expected, 2));
  EXPECT_EQ(expected, 1);  // Updated to actual
}

// 3. Standalone Fences

TEST(Fence, AcquireRelease) {
  int data = 0;
  std::atomic<int> flag{0};

  std::thread producer([&] {
    data = 100;
    std::atomic_thread_fence(std::memory_order_release);
    flag.store(1, std::memory_order_relaxed);
  });

  std::thread consumer([&] {
    while (flag.load(std::memory_order_relaxed) == 0);
    std::atomic_thread_fence(std::memory_order_acquire);
    EXPECT_EQ(data, 100);
  });

  producer.join();
  consumer.join();
}
