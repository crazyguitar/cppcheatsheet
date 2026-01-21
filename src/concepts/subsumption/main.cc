#include <gtest/gtest.h>

#include <concepts>
#include <string>

template <typename T>
concept Animal = requires(T t) { t.speak(); };

template <typename T>
concept Dog = Animal<T> && requires(T t) { t.bark(); };

std::string greet(Animal auto& a) {
  a.speak();
  return "animal";
}

std::string greet(Dog auto& d) {
  d.bark();
  return "dog";
}

struct Cat {
  void speak() {}
};

struct Beagle {
  void speak() {}
  void bark() {}
};

TEST(ConceptsSubsumption, MoreConstrainedWins) {
  Cat c;
  Beagle b;
  EXPECT_EQ(greet(c), "animal");
  EXPECT_EQ(greet(b), "dog");  // Dog subsumes Animal
}
