#include <gtest/gtest.h>

#include <utility>

TEST(TemplateLambda, ExplicitTemplate) {
  auto sum = []<typename... Args>(Args&&... args) { return (std::forward<Args>(args) + ...); };
  EXPECT_EQ(sum(1, 2, 3, 4, 5), 15);
}
