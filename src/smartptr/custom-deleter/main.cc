#include <gtest/gtest.h>

#include <memory>

TEST(CustomDeleter, UniquePtr) {
  bool deleted = false;
  {
    auto deleter = [&deleted](int* p) {
      deleted = true;
      delete p;
    };
    std::unique_ptr<int, decltype(deleter)> ptr(new int(42), deleter);
  }
  EXPECT_TRUE(deleted);
}

TEST(CustomDeleter, SharedPtr) {
  bool deleted = false;
  {
    auto deleter = [&deleted](int* p) {
      deleted = true;
      delete p;
    };
    std::shared_ptr<int> ptr(new int(42), deleter);
  }
  EXPECT_TRUE(deleted);
}
