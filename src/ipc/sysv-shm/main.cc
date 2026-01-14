#include <gtest/gtest.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(SysvShm, SharedMemoryCommunication) {
  key_t key = ftok("/tmp", 'T');
  int shmid = shmget(key, 4096, IPC_CREAT | 0666);
  ASSERT_NE(shmid, -1);
  char* ptr = (char*)shmat(shmid, NULL, 0);
  ASSERT_NE(ptr, (char*)-1);

  pid_t pid = fork();
  if (pid == 0) {
    strcpy(ptr, "sysv_shm_test");
    shmdt(ptr);
    _exit(0);
  } else {
    wait(NULL);
    EXPECT_STREQ(ptr, "sysv_shm_test");
    shmdt(ptr);
    shmctl(shmid, IPC_RMID, NULL);
  }
}
