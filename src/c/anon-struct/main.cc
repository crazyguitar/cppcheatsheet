#include <gtest/gtest.h>

struct vector3 {
  union {
    struct {
      float x, y, z;
    };
    float components[3];
  };
};

TEST(AnonStruct, DirectAccess) {
  vector3 v = {.x = 1.0f, .y = 2.0f, .z = 3.0f};

  EXPECT_FLOAT_EQ(v.x, 1.0f);
  EXPECT_FLOAT_EQ(v.y, 2.0f);
  EXPECT_FLOAT_EQ(v.z, 3.0f);

  EXPECT_FLOAT_EQ(v.components[0], 1.0f);
  EXPECT_FLOAT_EQ(v.components[1], 2.0f);
  EXPECT_FLOAT_EQ(v.components[2], 3.0f);
}
