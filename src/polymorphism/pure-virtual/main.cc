#include <gtest/gtest.h>

#include <type_traits>

struct Shape {
  virtual double area() const = 0;
  virtual ~Shape() = default;
};

struct Circle : Shape {
  explicit Circle(double r) : r_(r) {}
  double area() const override { return 3.14159 * r_ * r_; }

 private:
  double r_;
};

struct Square : Shape {
  explicit Square(double s) : s_(s) {}
  double area() const override { return s_ * s_; }

 private:
  double s_;
};

TEST(PureVirtual, AbstractClassIsNotInstantiable) {
  static_assert(std::is_abstract_v<Shape>, "Shape should be abstract");
  static_assert(!std::is_abstract_v<Circle>, "Circle should be concrete");
}

TEST(PureVirtual, DerivedDispatchesThroughBasePointer) {
  Circle c(2.0);
  Square s(3.0);
  Shape* shapes[] = {&c, &s};
  EXPECT_NEAR(shapes[0]->area(), 12.56636, 1e-4);
  EXPECT_NEAR(shapes[1]->area(), 9.0, 1e-9);
}
