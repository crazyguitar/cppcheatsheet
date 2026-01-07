#include <gtest/gtest.h>
#include <pthread.h>

namespace {
__thread int tls_var = 0;
int result1 = 0, result2 = 0;

void* worker(void* arg) {
  int id = *(int*)arg;
  tls_var = id;
  if (id == 1)
    result1 = tls_var;
  else
    result2 = tls_var;
  return NULL;
}

TEST(ThreadLocal, SeparateValues) {
  pthread_t t1, t2;
  int id1 = 1, id2 = 2;
  pthread_create(&t1, NULL, worker, &id1);
  pthread_create(&t2, NULL, worker, &id2);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  EXPECT_EQ(result1, 1);
  EXPECT_EQ(result2, 2);
}
}  // namespace
