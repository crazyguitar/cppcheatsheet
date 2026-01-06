#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>
#endif

TEST(EpollKqueueServer, Setup) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int on = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(15570);
  addr.sin_addr.s_addr = INADDR_ANY;
  bind(s, (struct sockaddr*)&addr, sizeof(addr));
  listen(s, 10);

#ifdef __linux__
  int eq = epoll_create1(0);
  struct epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.fd = s;
  epoll_ctl(eq, EPOLL_CTL_ADD, s, &ev);
  close(eq);
#elif defined(__APPLE__) || defined(__FreeBSD__)
  int kq = kqueue();
  struct kevent ev;
  EV_SET(&ev, s, EVFILT_READ, EV_ADD, 0, 0, NULL);
  kevent(kq, &ev, 1, NULL, 0, NULL);
  close(kq);
#endif
  close(s);
}
