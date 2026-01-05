#include <gtest/gtest.h>

#include <functional>
#include <queue>
#include <vector>

TEST(PriorityQueue, MaxHeap) {
  std::priority_queue<int> pq;
  for (int x : {1, 5, 2, 1, 3}) {
    pq.push(x);
  }

  EXPECT_EQ(pq.top(), 5);
  pq.pop();
  EXPECT_EQ(pq.top(), 3);
}

TEST(PriorityQueue, MinHeap) {
  std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
  for (int x : {1, 5, 2, 1, 3}) {
    pq.push(x);
  }

  EXPECT_EQ(pq.top(), 1);
  pq.pop();
  EXPECT_EQ(pq.top(), 1);
  pq.pop();
  EXPECT_EQ(pq.top(), 2);
}

TEST(PriorityQueue, CustomComparator) {
  auto cmp = [](int a, int b) { return a > b; };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

  for (int x : {1, 5, 2, 1, 3}) {
    pq.push(x);
  }

  EXPECT_EQ(pq.top(), 1);
}
