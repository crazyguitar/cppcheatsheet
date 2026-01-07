#include <gtest/gtest.h>
#include <pthread.h>

namespace {
int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* increment(void* arg) {
  for (int i = 0; i < 100000; i++) {
    pthread_mutex_lock(&lock);
    counter++;
    pthread_mutex_unlock(&lock);
  }
  return NULL;
}

TEST(Mutex, CounterSync) {
  counter = 0;
  pthread_t t1, t2;
  pthread_create(&t1, NULL, increment, NULL);
  pthread_create(&t2, NULL, increment, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  EXPECT_EQ(counter, 200000);
}
}  // namespace
