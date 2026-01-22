#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

// Moved-from objects must be in a valid but unspecified state:
// 1. Destructor can be safely called
// 2. Can be assigned a new value
// 3. Other operations have unspecified results

TEST(MovedFromState, DestructorSafe) {
  std::string s1 = "hello";
  std::string s2 = std::move(s1);

  // s1's destructor will be called at end of scope - must be safe
  // This test passes if no crash occurs
  EXPECT_EQ(s2, "hello");
}

TEST(MovedFromState, CanAssignNewValue) {
  std::string s1 = "hello";
  std::string s2 = std::move(s1);

  // Assigning to moved-from object is always valid
  s1 = "world";
  EXPECT_EQ(s1, "world");
}

TEST(MovedFromState, VectorMovedFrom) {
  std::vector<int> v1 = {1, 2, 3, 4, 5};
  std::vector<int> v2 = std::move(v1);

  // v1 is in valid but unspecified state
  // Safe operations:
  v1.clear();         // OK
  v1 = {10, 20, 30};  // OK: assign new value

  EXPECT_EQ(v1.size(), 3u);
  EXPECT_EQ(v2.size(), 5u);
}

// User-defined type with proper moved-from state
class Resource {
 public:
  explicit Resource(int value) : ptr_(new int(value)) {}

  ~Resource() { delete ptr_; }  // Safe even if ptr_ is nullptr

  Resource(Resource&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}

  Resource& operator=(Resource&& other) noexcept {
    if (this != &other) {
      delete ptr_;
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }

  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;

  bool has_value() const { return ptr_ != nullptr; }
  int value() const { return ptr_ ? *ptr_ : 0; }

 private:
  int* ptr_ = nullptr;
};

TEST(MovedFromState, UserDefinedType) {
  Resource r1(42);
  EXPECT_TRUE(r1.has_value());

  Resource r2 = std::move(r1);

  // r1 is in valid empty state
  EXPECT_FALSE(r1.has_value());
  EXPECT_TRUE(r2.has_value());
  EXPECT_EQ(r2.value(), 42);

  // Can assign new value to r1
  r1 = Resource(100);
  EXPECT_TRUE(r1.has_value());
  EXPECT_EQ(r1.value(), 100);
}
