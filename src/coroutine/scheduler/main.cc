#include <gtest/gtest.h>
#include <io/handle.h>
#include <io/io.h>

#include <thread>

struct TestHandle : Handle {
  int run_count = 0, stop_count = 0;
  void run() override { ++run_count; }
  void stop() override { ++stop_count; }
};

TEST(SchedulerTest, ImmediateScheduling) {
  TestHandle h;
  IO::Get().Call(h);
  EXPECT_EQ(h.GetState(), Handle::kScheduled);
  IO::Get().Run();
  EXPECT_EQ(h.run_count, 1);
  EXPECT_EQ(h.stop_count, 0);
}

TEST(SchedulerTest, Cancellation) {
  TestHandle h;
  IO::Get().Call(h);
  IO::Get().Cancel(h);
  IO::Get().Run();
  EXPECT_EQ(h.run_count, 0);
  EXPECT_EQ(h.stop_count, 1);
}

TEST(SchedulerTest, DelayedScheduling) {
  TestHandle h;
  IO::Get().Call(std::chrono::milliseconds(1), h);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  IO::Get().Run();
  EXPECT_EQ(h.run_count, 1);
}
