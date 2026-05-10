#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

// Track construction and move counts
struct Tracked {
  static int construct_count;
  static int move_count;
  static int copy_count;
  static void reset() { construct_count = move_count = copy_count = 0; }

  std::string data;

  explicit Tracked(const char* s) : data(s) { ++construct_count; }
  Tracked(const Tracked& o) : data(o.data) { ++copy_count; }
  Tracked(Tracked&& o) noexcept : data(std::move(o.data)) { ++move_count; }
};

int Tracked::construct_count = 0;
int Tracked::move_count = 0;
int Tracked::copy_count = 0;

TEST(Emplace, PushBackCreatesTemporaryAndMoves) {
  Tracked::reset();
  std::vector<Tracked> v;
  v.reserve(1);  // Prevent reallocation

  // push_back: creates temporary, then moves it into vector
  v.push_back(Tracked("hello"));  // 1 construction + 1 move

  EXPECT_EQ(Tracked::construct_count, 1);  // Temporary constructed
  EXPECT_EQ(Tracked::move_count, 1);       // Moved into vector
}

TEST(Emplace, EmplaceBackConstructsInPlace) {
  Tracked::reset();
  std::vector<Tracked> v;
  v.reserve(1);  // Prevent reallocation

  // emplace_back: constructs directly in vector's storage
  v.emplace_back("hello");  // 1 construction only

  EXPECT_EQ(Tracked::construct_count, 1);  // Constructed directly in vector
  EXPECT_EQ(Tracked::move_count, 0);       // No move needed
}

TEST(Emplace, PushBackCopiesLvalue) {
  Tracked::reset();
  std::vector<Tracked> v;
  v.reserve(2);  // Prevent reallocation

  Tracked t("hello");
  Tracked::reset();  // Reset after construction

  // push_back with lvalue: COPIES
  v.push_back(t);
  EXPECT_EQ(Tracked::copy_count, 1);
  EXPECT_EQ(Tracked::move_count, 0);

  // push_back with std::move: MOVES
  Tracked::reset();
  v.push_back(std::move(t));
  EXPECT_EQ(Tracked::copy_count, 0);
  EXPECT_EQ(Tracked::move_count, 1);
}

TEST(Emplace, ExpensiveTypesEmplace) {
  std::vector<std::vector<int>> vv;

  // emplace_back: constructs 1000-element vector directly in vv
  vv.emplace_back(1000, 0);  // Constructs vector<int>(1000, 0) in-place

  // push_back equivalent would be:
  // vv.push_back(std::vector<int>(1000, 0));  // construct + move

  EXPECT_EQ(vv.size(), 1u);
  EXPECT_EQ(vv[0].size(), 1000u);
  EXPECT_EQ(vv[0][0], 0);
}

TEST(Emplace, MapEmplaceAvoidsTemporaryPair) {
  std::map<std::string, std::string> m;

  // emplace: constructs pair<string,string> in-place
  auto [it, inserted] = m.emplace("key", "value");

  // insert equivalent would be:
  // m.insert(std::make_pair("key", "value"));  // construct pair + move

  EXPECT_TRUE(inserted);
  EXPECT_EQ(it->first, "key");
  EXPECT_EQ(it->second, "value");
}

// Non-copyable type - emplace avoids copy
struct NonCopyable {
  int value;
  explicit NonCopyable(int v) : value(v) {}
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) noexcept = default;
  NonCopyable& operator=(NonCopyable&&) noexcept = default;
};

TEST(Emplace, NonCopyableTypeWithEmplace) {
  std::vector<NonCopyable> v;
  v.reserve(2);  // Prevent reallocation

  v.emplace_back(42);  // OK: constructs in-place
  v.emplace_back(99);

  EXPECT_EQ(v[0].value, 42);
  EXPECT_EQ(v[1].value, 99);
}

// Caveat: emplace can hide implicit conversions
TEST(Emplace, CaveatImplicitConversion) {
  std::vector<std::vector<int>> vv;

  // This compiles but may not be intended!
  // emplace_back(10) calls vector<int>(10), creating vector of size 10
  vv.emplace_back(10);

  EXPECT_EQ(vv.size(), 1u);
  EXPECT_EQ(vv[0].size(), 10u);  // Not a vector containing {10}!

  // push_back would catch this:
  // vv.push_back(10);  // Compile error: no conversion from int
}
