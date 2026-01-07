#include <gtest/gtest.h>

const char* grade(int score) {
  switch (score) {
    case 90 ... 100:
      return "A";
    case 80 ... 89:
      return "B";
    case 70 ... 79:
      return "C";
    case 60 ... 69:
      return "D";
    case 0 ... 59:
      return "F";
    default:
      return "Invalid";
  }
}

TEST(CaseRange, GradesCorrectly) {
  EXPECT_STREQ(grade(95), "A");
  EXPECT_STREQ(grade(85), "B");
  EXPECT_STREQ(grade(73), "C");
  EXPECT_STREQ(grade(65), "D");
  EXPECT_STREQ(grade(45), "F");
}
