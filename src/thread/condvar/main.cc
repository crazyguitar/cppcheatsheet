#include <gtest/gtest.h>
#include <pthread.h>

namespace {
int ready = 0;
int consumed = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void* producer(void* arg) {
  pthread_mutex_lock(&lock);
  ready = 1;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&lock);
  return NULL;
}

void* consumer(void* arg) {
  pthread_mutex_lock(&lock);
  while (!ready) pthread_cond_wait(&cond, &lock);
  consumed = 1;
  pthread_mutex_unlock(&lock);
  return NULL;
}

TEST(CondVar, ProducerConsumer) {
  ready = 0;
  consumed = 0;
  pthread_t prod, cons;
  pthread_create(&cons, NULL, consumer, NULL);
  pthread_create(&prod, NULL, producer, NULL);
  pthread_join(prod, NULL);
  pthread_join(cons, NULL);
  EXPECT_EQ(consumed, 1);
}
}  // namespace
