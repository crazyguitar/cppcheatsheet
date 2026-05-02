#include <gtest/gtest.h>

#include <memory>

namespace virtual_dtor {

inline int& counter() {
  static int n = 0;
  return n;
}

struct BaseV {
  virtual ~BaseV() = default;
};

struct DerivedV : BaseV {
  ~DerivedV() noexcept override { ++counter(); }
};

}  // namespace virtual_dtor

TEST(VirtualDestructor, DerivedDestructorRunsThroughBasePointer) {
  using namespace virtual_dtor;
  counter() = 0;
  BaseV* p = new DerivedV;
  delete p;
  EXPECT_EQ(counter(), 1);
}

TEST(VirtualDestructor, UniquePtrToBaseAlsoCallsDerivedDtor) {
  using namespace virtual_dtor;
  counter() = 0;
  { std::unique_ptr<virtual_dtor::BaseV> p = std::make_unique<virtual_dtor::DerivedV>(); }
  EXPECT_EQ(counter(), 1);
}

// Without a virtual destructor, `delete base_ptr;` is undefined behavior when
// the dynamic type is a derived class. We do not exercise that path here
// because UB is not a stable test condition; sanitizers are the right tool.
