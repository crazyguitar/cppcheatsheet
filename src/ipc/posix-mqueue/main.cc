// Note: POSIX message queues not available on macOS
#include <fcntl.h>
#include <gtest/gtest.h>
#include <mqueue.h>
#include <sys/wait.h>

#include <cstring>

TEST(PosixMqueue, MessageQueueCommunication) {
  const char* name = "/test_mq";
  mq_unlink(name);

  struct mq_attr attr = {.mq_maxmsg = 10, .mq_msgsize = 256};
  mqd_t mq = mq_open(name, O_CREAT | O_RDWR, 0666, &attr);
  ASSERT_NE(mq, (mqd_t)-1);

  pid_t pid = fork();
  if (pid == 0) {
    mq_send(mq, "mq_test", 8, 1);
    mq_close(mq);
    _exit(0);
  } else {
    wait(NULL);
    char buf[256];
    unsigned int prio;
    ssize_t n = mq_receive(mq, buf, sizeof(buf), &prio);
    EXPECT_GT(n, 0);
    EXPECT_STREQ(buf, "mq_test");
    EXPECT_EQ(prio, 1u);
    mq_close(mq);
    mq_unlink(name);
  }
}
