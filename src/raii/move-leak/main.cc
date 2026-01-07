#include <gtest/gtest.h>

#include <utility>

class Resource {
 public:
  Resource() : value_(new int(42)) {}

  ~Resource() {
    delete value_;
    destructor_count++;
  }

  // Correct move: resets source
  Resource(Resource&& other) noexcept : value_(std::exchange(other.value_, nullptr)) { move_count++; }

  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;
  Resource& operator=(Resource&&) = delete;

  bool has_value() const { return value_ != nullptr; }

  static int destructor_count;
  static int move_count;
  static void reset_counts() { destructor_count = move_count = 0; }

 private:
  int* value_;
};

int Resource::destructor_count = 0;
int Resource::move_count = 0;

TEST(MoveLeak, MoveDoesNotCallDestructor) {
  Resource::reset_counts();
  {
    Resource r1;
    Resource r2(std::move(r1));
    EXPECT_EQ(Resource::move_count, 1);
    EXPECT_EQ(Resource::destructor_count, 0);  // No destructor called yet
  }
  // Both destructors called when scope ends
  EXPECT_EQ(Resource::destructor_count, 2);
}

TEST(MoveLeak, MovedFromObjectStillExists) {
  Resource r1;
  Resource r2(std::move(r1));

  // r1 still exists, but should be in empty state
  EXPECT_FALSE(r1.has_value());
  EXPECT_TRUE(r2.has_value());
}

TEST(MoveLeak, ProperResetPreventsDoubleFree) {
  Resource::reset_counts();
  {
    Resource r1;
    Resource r2(std::move(r1));
    // r1.value_ is nullptr, so destructor won't double-delete
  }
  // If we reach here without crash, the move was implemented correctly
  EXPECT_EQ(Resource::destructor_count, 2);
}
