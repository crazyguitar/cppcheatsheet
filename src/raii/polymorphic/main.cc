#include <gtest/gtest.h>

#include <string>

class Animal {
 public:
  Animal() = default;
  Animal(const Animal&) = delete;
  Animal& operator=(const Animal&) = delete;
  virtual ~Animal() = default;

  virtual std::string speak() { return "..."; }
};

class Dog : public Animal {
 public:
  std::string speak() override { return "woof"; }
};

class Cat : public Animal {
 public:
  std::string speak() override { return "meow"; }
};

TEST(Polymorphic, VirtualDispatch) {
  Dog dog;
  Cat cat;
  Animal* animals[] = {&dog, &cat};

  EXPECT_EQ(animals[0]->speak(), "woof");
  EXPECT_EQ(animals[1]->speak(), "meow");
}

TEST(Polymorphic, ReferencePreservesType) {
  Dog dog;
  Animal& ref = dog;
  EXPECT_EQ(ref.speak(), "woof");
}
