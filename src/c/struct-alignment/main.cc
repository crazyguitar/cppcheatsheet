#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>

struct Bad {
  char a;  // 1 byte + 3 padding
  int b;   // 4 bytes
  char c;  // 1 byte + 3 padding
};  // Total: 12 bytes

struct Good {
  int b;   // 4 bytes
  char a;  // 1 byte
  char c;  // 1 byte + 2 padding
};  // Total: 8 bytes

struct Packed {
  char a;
  int b;
  char c;
} __attribute__((packed));  // Total: 6 bytes

TEST(StructAlignment, SizeComparison) {
  printf("sizeof(Bad)    = %zu\n", sizeof(Bad));
  printf("sizeof(Good)   = %zu\n", sizeof(Good));
  printf("sizeof(Packed) = %zu\n", sizeof(Packed));

  printf("offsetof(Bad, a) = %zu\n", offsetof(Bad, a));
  printf("offsetof(Bad, b) = %zu\n", offsetof(Bad, b));
  printf("offsetof(Bad, c) = %zu\n", offsetof(Bad, c));

  EXPECT_EQ(sizeof(Bad), 12u);
  EXPECT_EQ(sizeof(Good), 8u);
  EXPECT_EQ(sizeof(Packed), 6u);

  EXPECT_EQ(offsetof(Bad, a), 0u);
  EXPECT_EQ(offsetof(Bad, b), 4u);
  EXPECT_EQ(offsetof(Bad, c), 8u);
}
