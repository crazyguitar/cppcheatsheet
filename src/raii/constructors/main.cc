#include <gtest/gtest.h>

#include <utility>

class Resource {
 public:
  Resource(int x) : x_(x) {}
  Resource() = default;

  Resource(const Resource& other) : Resource(other.x_) { copy_count++; }

  Resource& operator=(const Resource& other) {
    x_ = other.x_;
    copy_assign_count++;
    return *this;
  }

  Resource(Resource&& other) noexcept : x_(std::move(other.x_)) {
    move_count++;
    other.x_ = 0;
  }

  Resource& operator=(Resource&& other) noexcept {
    x_ = std::move(other.x_);
    move_assign_count++;
    return *this;
  }

  int value() const { return x_; }

  static int copy_count;
  static int copy_assign_count;
  static int move_count;
  static int move_assign_count;

  static void reset_counts() { copy_count = copy_assign_count = move_count = move_assign_count = 0; }

 private:
  int x_ = 0;
};

int Resource::copy_count = 0;
int Resource::copy_assign_count = 0;
int Resource::move_count = 0;
int Resource::move_assign_count = 0;

TEST(Constructors, CopyConstructor) {
  Resource::reset_counts();
  Resource r1(42);
  Resource r2(r1);
  EXPECT_EQ(r2.value(), 42);
  EXPECT_EQ(Resource::copy_count, 1);
}

TEST(Constructors, MoveConstructor) {
  Resource::reset_counts();
  Resource r1(42);
  Resource r2(std::move(r1));
  EXPECT_EQ(r2.value(), 42);
  EXPECT_EQ(r1.value(), 0);
  EXPECT_EQ(Resource::move_count, 1);
}

TEST(Constructors, CopyAssignment) {
  Resource::reset_counts();
  Resource r1(42), r2;
  r2 = r1;
  EXPECT_EQ(r2.value(), 42);
  EXPECT_EQ(Resource::copy_assign_count, 1);
}

TEST(Constructors, MoveAssignment) {
  Resource::reset_counts();
  Resource r1(42), r2;
  r2 = std::move(r1);
  EXPECT_EQ(r2.value(), 42);
  EXPECT_EQ(Resource::move_assign_count, 1);
}
