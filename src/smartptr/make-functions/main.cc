#include <gtest/gtest.h>

#include <memory>

struct Widget {
  int x = 0;
  int y = 0;
};

TEST(MakeFunctions, MakeUnique) {
  auto ptr = std::make_unique<Widget>();
  EXPECT_NE(ptr, nullptr);
}

TEST(MakeFunctions, MakeShared) {
  auto ptr = std::make_shared<Widget>();
  EXPECT_EQ(ptr.use_count(), 1);
}
