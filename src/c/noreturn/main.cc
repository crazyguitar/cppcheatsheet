#include <gtest/gtest.h>

[[noreturn]] void always_throws() { throw std::runtime_error("expected"); }

TEST(Noreturn, ThrowsException) { EXPECT_THROW(always_throws(), std::runtime_error); }
