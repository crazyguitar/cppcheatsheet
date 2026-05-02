#include <gtest/gtest.h>

#include <string>

namespace mi {

struct Drawable {
  virtual std::string draw() const = 0;
  virtual ~Drawable() = default;
};

struct Serializable {
  virtual std::string save() const = 0;
  virtual ~Serializable() = default;
};

struct Widget : Drawable, Serializable {
  std::string draw() const override { return "draw"; }
  std::string save() const override { return "save"; }
};

}  // namespace mi

TEST(MultipleInheritance, BothInterfacesDispatchToWidget) {
  mi::Widget w;
  mi::Drawable* d = &w;
  mi::Serializable* s = &w;
  EXPECT_EQ(d->draw(), "draw");
  EXPECT_EQ(s->save(), "save");
}

TEST(MultipleInheritance, BasePointersMayHaveDifferentAddresses) {
  mi::Widget w;
  void* widget_addr = static_cast<void*>(&w);
  void* drawable_addr = static_cast<void*>(static_cast<mi::Drawable*>(&w));
  void* serializable_addr = static_cast<void*>(static_cast<mi::Serializable*>(&w));
  // The first base often shares the widget's address; the second base sits
  // at a non-zero offset. We only assert that *some* offset exists for the
  // second base, since exact layout is ABI-defined.
  EXPECT_NE(drawable_addr, serializable_addr);
  EXPECT_TRUE(drawable_addr == widget_addr || serializable_addr == widget_addr);
}
