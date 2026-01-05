#include <gtest/gtest.h>

#include <cstring>
#include <utility>

class Buffer {
 public:
  Buffer(const char* data, size_t size) : data_(new char[size]), size_(size) { std::memcpy(data_, data, size); }

  ~Buffer() { delete[] data_; }

  Buffer(const Buffer& other) : Buffer(other.data_, other.size_) {}

  Buffer(Buffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  Buffer& operator=(const Buffer& other) { return *this = Buffer(other); }

  Buffer& operator=(Buffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }

  const char* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  char* data_ = nullptr;
  size_t size_ = 0;
};

TEST(RuleOfFive, MoveConstructor) {
  Buffer buf1("hello", 6);
  Buffer buf2(std::move(buf1));
  EXPECT_STREQ(buf2.data(), "hello");
  EXPECT_EQ(buf1.data(), nullptr);
}

TEST(RuleOfFive, MoveAssignment) {
  Buffer buf1("hello", 6);
  Buffer buf2("world", 6);
  buf2 = std::move(buf1);
  EXPECT_STREQ(buf2.data(), "hello");
  // After swap, buf1 has buf2's old data
  EXPECT_STREQ(buf1.data(), "world");
}

TEST(RuleOfFive, CopyAssignmentViaCopyAndSwap) {
  Buffer buf1("hello", 6);
  Buffer buf2("world", 6);
  buf2 = buf1;
  EXPECT_STREQ(buf2.data(), "hello");
  EXPECT_STREQ(buf1.data(), "hello");
}
