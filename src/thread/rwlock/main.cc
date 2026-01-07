#include <gtest/gtest.h>
#include <pthread.h>

namespace {
int data = 0;
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
int read_val1 = -1, read_val2 = -1;

void* reader(void* arg) {
  int* out = (int*)arg;
  pthread_rwlock_rdlock(&rwlock);
  *out = data;
  pthread_rwlock_unlock(&rwlock);
  return NULL;
}

void* writer(void* arg) {
  pthread_rwlock_wrlock(&rwlock);
  data = 42;
  pthread_rwlock_unlock(&rwlock);
  return NULL;
}

TEST(RWLock, ReadWrite) {
  data = 0;
  pthread_t w, r1, r2;
  pthread_create(&w, NULL, writer, NULL);
  pthread_join(w, NULL);

  pthread_create(&r1, NULL, reader, &read_val1);
  pthread_create(&r2, NULL, reader, &read_val2);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);

  EXPECT_EQ(read_val1, 42);
  EXPECT_EQ(read_val2, 42);
}
}  // namespace
