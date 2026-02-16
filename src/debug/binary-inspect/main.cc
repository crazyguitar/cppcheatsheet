// Binary Inspection Example
// Compile and use binary tools to inspect:
//
//   nm:       nm binary-inspect_test | c++filt
//   readelf:  readelf -h binary-inspect_test
//   objdump:  objdump -d -C binary-inspect_test
//   strings:  strings binary-inspect_test
//   size:     size binary-inspect_test
//   file:     file binary-inspect_test
//   ldd:      ldd binary-inspect_test

#include <gtest/gtest.h>

#include "helper.h"

static const char *version_string = "binary-inspect v1.0";

namespace demo {
class Widget {
  int val_;

 public:
  Widget(int v) : val_(v) {}
  int value() const { return val_; }
};
}  // namespace demo

TEST(BinaryInspect, Symbols) {
  // global_initialized lives in .data (initialized data)
  EXPECT_EQ(global_initialized, 42);

  // global_uninitialized lives in .bss (zero-initialized)
  EXPECT_EQ(global_uninitialized, 0);

  // add() is an external function in .text
  EXPECT_EQ(add(2, 3), 5);

  // square() is inlined from header
  EXPECT_EQ(square(4), 16);

  // C++ mangled symbol
  demo::Widget w(10);
  EXPECT_EQ(w.value(), 10);

  // version_string is in .rodata
  EXPECT_STREQ(version_string, "binary-inspect v1.0");
}
