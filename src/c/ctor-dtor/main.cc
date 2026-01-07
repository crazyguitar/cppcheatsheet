#include <gtest/gtest.h>

static int init_called = 0;
static int cleanup_called = 0;

__attribute__((constructor))
static void init() {
  init_called = 1;
}

__attribute__((destructor))
static void cleanup() {
  cleanup_called = 1;
}

TEST(CtorDtor, ConstructorCalled) {
  EXPECT_EQ(init_called, 1);
}
