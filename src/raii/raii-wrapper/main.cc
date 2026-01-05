#include <gtest/gtest.h>

#include <cstdio>
#include <stdexcept>
#include <utility>

class File {
 public:
  explicit File(const char* path, const char* mode) : handle_(std::fopen(path, mode)) {
    if (!handle_) {
      throw std::runtime_error("Failed to open file");
    }
  }

  ~File() {
    if (handle_) {
      std::fclose(handle_);
    }
  }

  File(const File&) = delete;
  File& operator=(const File&) = delete;

  File(File&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

  File& operator=(File&& other) noexcept {
    if (this != &other) {
      if (handle_) std::fclose(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  std::FILE* get() const { return handle_; }
  bool is_open() const { return handle_ != nullptr; }

 private:
  std::FILE* handle_;
};

TEST(RAIIWrapper, FileOpensAndCloses) {
  {
    File f("/tmp/raii_test.txt", "w");
    EXPECT_TRUE(f.is_open());
    std::fputs("test", f.get());
  }
  // File closed when f goes out of scope
  std::remove("/tmp/raii_test.txt");
}

TEST(RAIIWrapper, MoveTransfersOwnership) {
  File f1("/tmp/raii_test2.txt", "w");
  File f2(std::move(f1));
  EXPECT_FALSE(f1.is_open());
  EXPECT_TRUE(f2.is_open());
  std::remove("/tmp/raii_test2.txt");
}

TEST(RAIIWrapper, ThrowsOnInvalidFile) { EXPECT_THROW(File("/nonexistent/path/file.txt", "r"), std::runtime_error); }
