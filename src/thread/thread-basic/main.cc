#include <gtest/gtest.h>
#include <pthread.h>

namespace {
int result1 = 0, result2 = 0;

void* worker(void* arg) {
  int id = *(int*)arg;
  if (id == 1)
    result1 = id;
  else
    result2 = id;
  return NULL;
}

TEST(ThreadBasic, CreateAndJoin) {
  pthread_t t1, t2;
  int id1 = 1, id2 = 2;

  ASSERT_EQ(pthread_create(&t1, NULL, worker, &id1), 0);
  ASSERT_EQ(pthread_create(&t2, NULL, worker, &id2), 0);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);

  EXPECT_EQ(result1, 1);
  EXPECT_EQ(result2, 2);
}
}  // namespace
