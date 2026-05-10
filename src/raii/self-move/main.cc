#include <gtest/gtest.h>

#include <cstring>
#include <utility>

// Safe self-move using explicit check
class BufferWithCheck {
 public:
  explicit BufferWithCheck(size_t size) : data_(new char[size]()), size_(size) {}

  ~BufferWithCheck() { delete[] data_; }

  BufferWithCheck(BufferWithCheck&& other) noexcept : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  BufferWithCheck& operator=(BufferWithCheck&& other) noexcept {
    if (this != &other) {  // Explicit self-check
      delete[] data_;
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  BufferWithCheck(const BufferWithCheck&) = delete;
  BufferWithCheck& operator=(const BufferWithCheck&) = delete;

  size_t size() const { return size_; }
  bool valid() const { return data_ != nullptr || size_ == 0; }

 private:
  char* data_ = nullptr;
  size_t size_ = 0;
};

// Safe self-move using swap (handles self-move automatically)
class BufferWithSwap {
 public:
  explicit BufferWithSwap(size_t size) : data_(new char[size]()), size_(size) {}

  ~BufferWithSwap() { delete[] data_; }

  BufferWithSwap(BufferWithSwap&& other) noexcept : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  BufferWithSwap& operator=(BufferWithSwap&& other) noexcept {
    std::swap(data_, other.data_);  // No self-check needed
    std::swap(size_, other.size_);
    return *this;
    // other's destructor cleans up our old data
  }

  BufferWithSwap(const BufferWithSwap&) = delete;
  BufferWithSwap& operator=(const BufferWithSwap&) = delete;

  size_t size() const { return size_; }
  bool valid() const { return data_ != nullptr || size_ == 0; }

 private:
  char* data_ = nullptr;
  size_t size_ = 0;
};

TEST(SelfMove, ExplicitCheckHandlesSelfMove) {
  BufferWithCheck b(100);
  BufferWithCheck* p = &b;

  b = std::move(*p);  // Self-move through pointer

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.size(), 100u);  // Unchanged due to self-check
}

TEST(SelfMove, SwapHandlesSelfMove) {
  BufferWithSwap b(100);
  BufferWithSwap* p = &b;

  b = std::move(*p);  // Self-move through pointer

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.size(), 100u);  // Swap with self is no-op
}

TEST(SelfMove, NormalMoveStillWorks) {
  BufferWithCheck b1(100);
  BufferWithCheck b2(200);

  b1 = std::move(b2);

  EXPECT_EQ(b1.size(), 200u);
  EXPECT_EQ(b2.size(), 0u);
}
