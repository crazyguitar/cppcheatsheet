#include <gtest/gtest.h>

#include <mutex>
#include <thread>
#include <vector>

// std::lock_guard / std::unique_lock / std::scoped_lock are the standard
// library's RAII wrappers for mutexes: acquire on construction, release in
// the destructor. They make locking exception-safe (the mutex is always
// released, even if the protected scope throws).

TEST(LockGuard, ScopedMutualExclusion) {
  std::mutex m;
  int counter = 0;
  auto worker = [&] {
    for (int i = 0; i < 10000; ++i) {
      std::lock_guard<std::mutex> guard(m);
      ++counter;
    }
  };
  std::thread t1(worker), t2(worker);
  t1.join();
  t2.join();
  EXPECT_EQ(counter, 20000);
}

TEST(UniqueLock, DeferredAcquisitionAndManualRelease) {
  std::mutex m;
  std::unique_lock<std::mutex> lock(m, std::defer_lock);
  EXPECT_FALSE(lock.owns_lock());
  lock.lock();
  EXPECT_TRUE(lock.owns_lock());
  lock.unlock();  // Manually release early; destructor handles the rest.
  EXPECT_FALSE(lock.owns_lock());
}

TEST(ScopedLock, LocksMultipleMutexesWithoutDeadlock) {
  std::mutex m1, m2;
  // std::scoped_lock (C++17) uses a deadlock-avoidance algorithm when
  // locking two or more mutexes. Safer than hand-ordered lock_guard pairs.
  {
    std::scoped_lock lock(m1, m2);
    // Probe from a different thread: calling try_lock on the thread that
    // already owns a non-recursive std::mutex is undefined behavior, so
    // mutual exclusion must be verified from outside the owning thread.
    std::thread probe([&] {
      EXPECT_FALSE(m1.try_lock());
      EXPECT_FALSE(m2.try_lock());
    });
    probe.join();
  }
  // After the scoped_lock's destructor, both mutexes are released.
  EXPECT_TRUE(m1.try_lock());
  EXPECT_TRUE(m2.try_lock());
  m1.unlock();
  m2.unlock();
}
