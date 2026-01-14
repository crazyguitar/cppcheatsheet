#include <gtest/gtest.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(PosixSemaphore, ProcessSynchronization) {
  const char* name = "/test_sem";
  sem_unlink(name);

  sem_t* sem = sem_open(name, O_CREAT, 0666, 0);
  ASSERT_NE(sem, SEM_FAILED);

  pid_t pid = fork();
  if (pid == 0) {
    usleep(10000);  // 10ms delay
    sem_post(sem);
    sem_close(sem);
    _exit(0);
  } else {
    sem_wait(sem);  // Blocks until child posts
    int status;
    waitpid(pid, &status, WNOHANG);
    // Child should have exited or be exiting
    sem_close(sem);
    sem_unlink(name);
    wait(NULL);
    SUCCEED();
  }
}
