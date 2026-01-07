#include <gtest/gtest.h>

constexpr int BUFFER_SIZE = 1024;
constexpr int DOUBLED = BUFFER_SIZE * 2;

TEST(Constexpr, CompileTimeConstants) {
  constexpr int local_const = 42;

  char buffer[BUFFER_SIZE];
  int arr[local_const];

  EXPECT_EQ(BUFFER_SIZE, 1024);
  EXPECT_EQ(DOUBLED, 2048);
  EXPECT_EQ(sizeof(buffer), 1024u);
  EXPECT_EQ(sizeof(arr), 42 * sizeof(int));
}
