#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace te {

// Value-semantic, type-erased wrapper. Anything with a `.draw() -> string`
// method satisfies the contract — no inheritance required.
class AnyDrawable {
  struct Concept {
    virtual ~Concept() = default;
    virtual std::string do_draw() const = 0;
    virtual std::unique_ptr<Concept> clone() const = 0;
  };
  template <typename T>
  struct Model : Concept {
    T value;
    explicit Model(T v) : value(std::move(v)) {}
    std::string do_draw() const override { return value.draw(); }
    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }
  };
  std::unique_ptr<Concept> p_;

 public:
  template <typename T>
  AnyDrawable(T x) : p_(std::make_unique<Model<T>>(std::move(x))) {}

  AnyDrawable(const AnyDrawable& other) : p_(other.p_->clone()) {}
  AnyDrawable(AnyDrawable&&) noexcept = default;
  AnyDrawable& operator=(AnyDrawable other) noexcept {
    p_ = std::move(other.p_);
    return *this;
  }

  std::string draw() const { return p_->do_draw(); }
};

// Two unrelated types — no shared base class.
struct Circle {
  double r;
  std::string draw() const { return "circle(" + std::to_string(int(r)) + ")"; }
};

struct Square {
  double s;
  std::string draw() const { return "square(" + std::to_string(int(s)) + ")"; }
};

}  // namespace te

TEST(TypeErasure, HeterogeneousValuesShareOneInterface) {
  std::vector<te::AnyDrawable> shapes;
  shapes.emplace_back(te::Circle{1.0});
  shapes.emplace_back(te::Square{2.0});
  EXPECT_EQ(shapes[0].draw(), "circle(1)");
  EXPECT_EQ(shapes[1].draw(), "square(2)");
}

TEST(TypeErasure, ValueSemanticsCopiesIndependently) {
  te::AnyDrawable a = te::Circle{1.0};
  te::AnyDrawable b = a;  // deep copy via Concept::clone
  EXPECT_EQ(a.draw(), "circle(1)");
  EXPECT_EQ(b.draw(), "circle(1)");
}

TEST(TypeErasure, ThirdPartyTypeWithoutModification) {
  // A type defined elsewhere with no knowledge of AnyDrawable still works,
  // as long as it has a `draw()` method.
  struct Triangle {
    std::string draw() const { return "triangle"; }
  };
  te::AnyDrawable t = Triangle{};
  EXPECT_EQ(t.draw(), "triangle");
}

TEST(TypeErasure, StdFunctionIsAStandardLibraryTypeErasedWrapper) {
  // std::function erases the call signature. Lambdas of different captures
  // and free functions all fit the same value type.
  std::vector<std::function<int(int)>> fs;
  fs.emplace_back([](int x) { return x + 1; });
  fs.emplace_back([k = 10](int x) { return x + k; });
  EXPECT_EQ(fs[0](1), 2);
  EXPECT_EQ(fs[1](1), 11);
}
