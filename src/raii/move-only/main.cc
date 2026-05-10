#include <gtest/gtest.h>

#include <cstdio>
#include <utility>
#include <vector>

// Move-only type: can be moved but not copied
class UniqueFile {
 public:
  explicit UniqueFile(const char* path) : handle_(std::fopen(path, "w")) {}

  ~UniqueFile() {
    if (handle_) std::fclose(handle_);
  }

  // Delete copy operations
  UniqueFile(const UniqueFile&) = delete;
  UniqueFile& operator=(const UniqueFile&) = delete;

  // Provide move operations
  UniqueFile(UniqueFile&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

  UniqueFile& operator=(UniqueFile&& other) noexcept {
    if (this != &other) {
      if (handle_) std::fclose(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  bool is_open() const { return handle_ != nullptr; }
  std::FILE* get() const { return handle_; }

 private:
  std::FILE* handle_ = nullptr;
};

TEST(MoveOnly, CannotCopy) {
  UniqueFile f1("/tmp/move_only_test1.txt");

  // These would fail to compile:
  // UniqueFile f2 = f1;           // Error: copy constructor deleted
  // UniqueFile f3(f1);            // Error: copy constructor deleted

  EXPECT_TRUE(f1.is_open());
  std::remove("/tmp/move_only_test1.txt");
}

TEST(MoveOnly, CanMove) {
  UniqueFile f1("/tmp/move_only_test2.txt");
  EXPECT_TRUE(f1.is_open());

  UniqueFile f2 = std::move(f1);  // OK: move constructor
  EXPECT_FALSE(f1.is_open());
  EXPECT_TRUE(f2.is_open());

  std::remove("/tmp/move_only_test2.txt");
}

TEST(MoveOnly, WorksWithContainers) {
  std::vector<UniqueFile> files;

  // Must use emplace_back or std::move
  files.emplace_back("/tmp/move_only_test3.txt");

  UniqueFile f("/tmp/move_only_test4.txt");
  files.push_back(std::move(f));  // OK: move into vector

  EXPECT_EQ(files.size(), 2u);
  EXPECT_TRUE(files[0].is_open());
  EXPECT_TRUE(files[1].is_open());

  std::remove("/tmp/move_only_test3.txt");
  std::remove("/tmp/move_only_test4.txt");
}
