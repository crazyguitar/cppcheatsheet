#include <gtest/gtest.h>
#include <io/result.h>

#include <stdexcept>

TEST(ResultTest, Value) {
  Result<int> r;
  EXPECT_FALSE(r.has_value());
  r.set_value(42);
  EXPECT_TRUE(r.has_value());
  EXPECT_EQ(r.result(), 42);
}

TEST(ResultTest, Exception) {
  Result<int> r;
  r.set_exception(std::make_exception_ptr(std::runtime_error("error")));
  EXPECT_THROW(r.result(), std::runtime_error);
}

TEST(ResultTest, Void) {
  Result<void> r;
  EXPECT_FALSE(r.has_value());
  r.return_void();
  EXPECT_TRUE(r.has_value());
  EXPECT_NO_THROW(r.result());
}
