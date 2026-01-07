#include <gtest/gtest.h>

struct Point {
  int x;
  int y;
};

TEST(ReinterpretCast, StructToBytes) {
  Point p{1, 2};
  char* buf = reinterpret_cast<char*>(&p);
  // First byte of x should be 1 (little-endian)
  EXPECT_EQ(buf[0], 1);
}

TEST(ReinterpretCast, IntToPointer) {
  int arr[] = {10, 20, 30};
  auto addr = reinterpret_cast<uintptr_t>(arr);
  int* ptr = reinterpret_cast<int*>(addr);
  EXPECT_EQ(ptr[0], 10);
}
