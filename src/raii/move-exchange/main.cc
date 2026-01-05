#include <gtest/gtest.h>

#include <cstdio>
#include <utility>

class File {
 public:
  explicit File(const char* path) : handle_(std::fopen(path, "w")) {}

  ~File() {
    if (handle_) std::fclose(handle_);
  }

  File(File&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

  File& operator=(File&& other) noexcept {
    if (this != &other) {
      if (handle_) std::fclose(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  File(const File&) = delete;
  File& operator=(const File&) = delete;

  bool is_open() const { return handle_ != nullptr; }
  std::FILE* get() const { return handle_; }

 private:
  std::FILE* handle_ = nullptr;
};

TEST(MoveExchange, MoveConstructorTransfersOwnership) {
  File f1("/tmp/exchange_test1.txt");
  EXPECT_TRUE(f1.is_open());

  File f2(std::move(f1));
  EXPECT_FALSE(f1.is_open());
  EXPECT_TRUE(f2.is_open());

  std::remove("/tmp/exchange_test1.txt");
}

TEST(MoveExchange, MoveAssignmentTransfersOwnership) {
  File f1("/tmp/exchange_test2.txt");
  File f2("/tmp/exchange_test3.txt");

  f2 = std::move(f1);
  EXPECT_FALSE(f1.is_open());
  EXPECT_TRUE(f2.is_open());

  std::remove("/tmp/exchange_test2.txt");
  std::remove("/tmp/exchange_test3.txt");
}

TEST(MoveExchange, SelfAssignmentSafe) {
  File f1("/tmp/exchange_test4.txt");
  f1 = std::move(f1);
  EXPECT_TRUE(f1.is_open());

  std::remove("/tmp/exchange_test4.txt");
}
