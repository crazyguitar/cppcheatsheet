#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>

TEST(UnixSocket, BidirectionalCommunication) {
  const char* path = "/tmp/test_sock";
  unlink(path);

  pid_t pid = fork();
  if (pid == 0) {
    usleep(50000);  // Wait for server
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    strcpy(addr.sun_path, path);
    connect(fd, (struct sockaddr*)&addr, sizeof(addr));
    write(fd, "request", 7);
    char buf[100];
    read(fd, buf, sizeof(buf));
    close(fd);
    _exit(strncmp(buf, "response", 8) == 0 ? 0 : 1);
  } else {
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    strcpy(addr.sun_path, path);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);

    int client = accept(srv, NULL, NULL);
    char buf[100];
    ssize_t n = read(client, buf, sizeof(buf));
    EXPECT_EQ(n, 7);
    EXPECT_EQ(strncmp(buf, "request", 7), 0);
    write(client, "response", 8);

    close(client);
    close(srv);
    int status;
    waitpid(pid, &status, 0);
    unlink(path);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
}
