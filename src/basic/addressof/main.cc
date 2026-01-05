#include <gtest/gtest.h>

#include <memory>

struct Widget {
  int value;
};

const Widget* operator&(const Widget& w) { return std::addressof(w); }

TEST(Addressof, OverloadedOperator) {
  Widget w;
  EXPECT_EQ(&w, std::addressof(w));
}
