#include <gtest/gtest.h>

#include <cstring>

namespace {
void duff_copy(char* to, const char* from, int count) {
  if (count <= 0) return;
  int n = (count + 7) / 8;
  switch (count % 8) {
    case 0:
      do {
        *to++ = *from++;
        case 7:
          *to++ = *from++;
        case 6:
          *to++ = *from++;
        case 5:
          *to++ = *from++;
        case 4:
          *to++ = *from++;
        case 3:
          *to++ = *from++;
        case 2:
          *to++ = *from++;
        case 1:
          *to++ = *from++;
      } while (--n > 0);
  }
}
}  // namespace

TEST(DuffsDevice, CopiesBytes) {
  const char src[] = "Hello, Duff!";
  char dst[20] = {0};
  duff_copy(dst, src, sizeof(src));
  EXPECT_STREQ(dst, src);
}
