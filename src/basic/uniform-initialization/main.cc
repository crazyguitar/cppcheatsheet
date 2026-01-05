#include <gtest/gtest.h>

#include <initializer_list>

class Widget {
 public:
  Widget(int a, double b) : used_initializer_list(false) {}
  Widget(std::initializer_list<long double> il) : used_initializer_list(true) {}
  bool used_initializer_list;
};

TEST(UniformInitialization, PrefersInitializerList) {
  Widget w{10, 5.0};
  EXPECT_TRUE(w.used_initializer_list);
}
