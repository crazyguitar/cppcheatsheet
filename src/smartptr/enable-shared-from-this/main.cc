#include <gtest/gtest.h>

#include <memory>

class Widget : public std::enable_shared_from_this<Widget> {
 public:
  std::shared_ptr<Widget> get_shared() { return shared_from_this(); }
};

TEST(EnableSharedFromThis, SharesOwnership) {
  auto w = std::make_shared<Widget>();
  auto w2 = w->get_shared();
  EXPECT_EQ(w.use_count(), 2);
}
