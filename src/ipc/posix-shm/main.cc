#include <gtest/gtest.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(PosixShm, SharedMemoryCommunication) {
  const char* name = "/test_shm";
  const size_t size = 4096;

  shm_unlink(name);
  int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, size), 0);
  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);
  close(fd);

  pid_t pid = fork();
  if (pid == 0) {
    strcpy((char*)ptr, "shared_data");
    _exit(0);
  } else {
    wait(NULL);
    EXPECT_STREQ((char*)ptr, "shared_data");
    munmap(ptr, size);
    shm_unlink(name);
  }
}
