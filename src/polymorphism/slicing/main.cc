#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace slicing {

struct Animal {
  virtual std::string speak() const { return "..."; }
  virtual ~Animal() = default;
};

struct Dog : Animal {
  std::string speak() const override { return "Woof"; }
};

inline std::string by_value(Animal a) { return a.speak(); }
inline std::string by_reference(const Animal& a) { return a.speak(); }

}  // namespace slicing

TEST(Slicing, PassByValueSlicesDerived) {
  slicing::Dog d;
  EXPECT_EQ(slicing::by_value(d), "...");
}

TEST(Slicing, PassByReferencePreservesDynamicType) {
  slicing::Dog d;
  EXPECT_EQ(slicing::by_reference(d), "Woof");
}

TEST(Slicing, VectorOfBaseValuesSlices) {
  std::vector<slicing::Animal> animals;
  animals.push_back(slicing::Dog{});
  EXPECT_EQ(animals[0].speak(), "...");
}

TEST(Slicing, VectorOfPointersPreservesPolymorphism) {
  std::vector<std::unique_ptr<slicing::Animal>> animals;
  animals.push_back(std::make_unique<slicing::Dog>());
  EXPECT_EQ(animals[0]->speak(), "Woof");
}
