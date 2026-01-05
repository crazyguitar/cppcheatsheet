#include <gtest/gtest.h>

#include <cstring>
#include <memory>

class Buffer {
 public:
  Buffer(const char* data, size_t size) : data_(new char[size]), size_(size) { std::memcpy(data_, data, size); }

  ~Buffer() { delete[] data_; }

  Buffer(const Buffer& other) : Buffer(other.data_, other.size_) {}

  Buffer& operator=(const Buffer& other) {
    if (this == std::addressof(other)) {
      return *this;
    }
    char* new_data = new char[other.size_];
    std::memcpy(new_data, other.data_, other.size_);
    delete[] data_;
    data_ = new_data;
    size_ = other.size_;
    return *this;
  }

  const char* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  char* data_;
  size_t size_;
};

TEST(RuleOfThree, CopyConstructor) {
  Buffer buf1("hello", 6);
  Buffer buf2(buf1);
  EXPECT_STREQ(buf2.data(), "hello");
  EXPECT_NE(buf1.data(), buf2.data());
}

TEST(RuleOfThree, CopyAssignment) {
  Buffer buf1("hello", 6);
  Buffer buf2("world", 6);
  buf2 = buf1;
  EXPECT_STREQ(buf2.data(), "hello");
}

TEST(RuleOfThree, SelfAssignment) {
  Buffer buf("hello", 6);
  Buffer* p = &buf;
  buf = *p;  // Self-assignment through pointer
  EXPECT_STREQ(buf.data(), "hello");
}
