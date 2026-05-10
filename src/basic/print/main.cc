#include <gtest/gtest.h>

#include <print>
#include <sstream>

TEST(Print, HelloWorld) {
  std::ostringstream oss;
  std::print(oss, "Hello, World!");
  EXPECT_EQ(oss.str(), "Hello, World!");
}

TEST(Println, HelloWorld) {
  std::ostringstream oss;
  std::println(oss, "Hello, World!");
  EXPECT_EQ(oss.str(), "Hello, World!\n");
}
