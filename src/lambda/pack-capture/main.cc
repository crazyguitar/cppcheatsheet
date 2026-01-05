#include <gtest/gtest.h>

#include <utility>

template <typename... Args>
auto make_delayed(Args&&... args) {
  return [... args = std::forward<Args>(args)]() { return (args + ...); };
}

TEST(PackCapture, DelayedCall) {
  auto delayed = make_delayed(1, 2, 3, 4, 5);
  EXPECT_EQ(delayed(), 15);
}
