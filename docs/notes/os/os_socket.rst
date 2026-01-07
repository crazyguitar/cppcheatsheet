================
Socket Reference
================

.. meta::
   :description: C socket programming tutorial with examples for TCP/UDP servers, DNS resolution with gethostbyname and getaddrinfo, byte order conversion, I/O multiplexing using select, poll, epoll, kqueue, and Unix domain sockets.
   :keywords: C socket programming, TCP server example, UDP server example, socket tutorial, select poll epoll kqueue, gethostbyname, getaddrinfo, Unix domain socket, network programming C, Berkeley sockets, htons htonl, inet_pton inet_ntop, socket bind listen accept

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Socket programming is the foundation of network communication in Unix-like
operating systems. The Berkeley sockets API, introduced in BSD 4.2 (1983),
has become the de facto standard for network programming across all major
platforms including Linux, macOS, Windows, and embedded systems.

At its core, a socket is an endpoint for communication—a file descriptor
that represents a connection to another process, either on the same machine
or across a network. The API provides a unified interface for multiple
protocols (TCP, UDP, Unix domain sockets) and address families (IPv4, IPv6).

DNS Lookup with ``gethostbyname``
---------------------------------

:Source: `src/socket/gethostbyname <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/gethostbyname>`_

The ``gethostbyname`` function resolves a hostname to one or more IP addresses.
While this function is deprecated in favor of the more modern ``getaddrinfo``,
it remains widely used in legacy codebases and is simpler for basic IPv4-only
lookups. The function returns a pointer to a ``hostent`` structure containing
the canonical hostname, alias names, address type, address length, and a
null-terminated list of network addresses. Note that ``gethostbyname`` is not
thread-safe; use ``getaddrinfo`` for concurrent applications.

.. code-block:: c

    #include <stdio.h>
    #include <netdb.h>
    #include <arpa/inet.h>

    int main(int argc, char *argv[]) {
      struct hostent *h = gethostbyname(argv[1]);
      if (!h) return 1;

      printf("Host: %s\n", h->h_name);
      for (struct in_addr **p = (struct in_addr **)h->h_addr_list; *p; p++)
        printf("IP: %s\n", inet_ntoa(**p));
    }

.. code-block:: bash

    $ ./gethostbyname www.google.com
    Host: www.google.com
    IP: 142.250.80.100

Byte Order Conversion
---------------------

:Source: `src/socket/byteorder <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/byteorder>`_

Network protocols universally use big-endian (network) byte order for
transmitting multi-byte integers, while most modern processors (x86, ARM)
use little-endian (host) byte order. The ``htons`` (host-to-network-short)
and ``htonl`` (host-to-network-long) functions convert 16-bit and 32-bit
values from host to network byte order respectively. The inverse functions
``ntohs`` and ``ntohl`` convert from network to host byte order. These
conversions are essential for writing portable network code that works
correctly on both big-endian and little-endian machines. On big-endian
systems, these functions are no-ops.

.. code-block:: c

    #include <stdio.h>
    #include <arpa/inet.h>

    int main(void) {
      uint16_t port = 8080;
      uint32_t addr = 0x7f000001;  // 127.0.0.1

      printf("host port: 0x%x -> network: 0x%x\n", port, htons(port));
      printf("host addr: 0x%x -> network: 0x%x\n", addr, htonl(addr));
    }

.. code-block:: bash

    $ ./byteorder
    host port: 0x1f90 -> network: 0x901f
    host addr: 0x7f000001 -> network: 0x100007f

Basic TCP Server
----------------

:Source: `src/socket/tcp-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/tcp-server>`_

A TCP server follows the classic socket-bind-listen-accept pattern. First,
create a socket with ``socket()``, specifying ``AF_INET`` for IPv4 and
``SOCK_STREAM`` for TCP. Setting ``SO_REUSEADDR`` allows the server to
restart immediately without waiting for the TIME_WAIT state to expire.
The ``bind()`` call associates the socket with a local address and port,
while ``listen()`` marks it as a passive socket ready to accept connections.
The ``accept()`` call blocks until a client connects, returning a new socket
descriptor for that specific connection. This simple echo server receives
data and sends it back to the client before closing the connection.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    int main(void) {
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int on = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

      for (;;) {
        int c = accept(s, NULL, NULL);
        char buf[1024] = {0};
        recv(c, buf, sizeof(buf) - 1, 0);
        send(c, buf, strlen(buf), 0);
        close(c);
      }
    }

.. code-block:: bash

    $ ./tcp-server &
    $ echo "Hello" | nc localhost 5566
    Hello

Basic UDP Server
----------------

:Source: `src/socket/udp-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/udp-server>`_

UDP (User Datagram Protocol) is a connectionless protocol, meaning there is
no handshake or persistent connection between client and server. Instead of
``accept()``, ``recv()``, and ``send()``, UDP servers use ``recvfrom()`` and
``sendto()`` which include the remote address as a parameter. Each datagram
is independent and contains the sender's address, allowing the server to
respond without maintaining any connection state. This makes UDP ideal for
applications where low latency is more important than guaranteed delivery,
such as DNS queries, streaming media, or online gaming. The trade-off is
that UDP provides no built-in reliability, ordering, or flow control.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    int main(void) {
      int s = socket(AF_INET, SOCK_DGRAM, 0);
      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));

      for (;;) {
        char buf[1024] = {0};
        struct sockaddr_in client;
        socklen_t len = sizeof(client);
        ssize_t n = recvfrom(s, buf, sizeof(buf), 0,
                             (struct sockaddr *)&client, &len);
        sendto(s, buf, n, 0, (struct sockaddr *)&client, len);
      }
    }

.. code-block:: bash

    $ ./udp-server &
    $ echo "Hello" | nc -u localhost 5566
    Hello

I/O Multiplexing with ``select``
--------------------------------

:Source: `src/socket/select-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/select-server>`_

The ``select`` system call enables a single-threaded server to monitor multiple
file descriptors simultaneously, waiting until one or more become ready for
I/O operations. This is the oldest and most portable I/O multiplexing mechanism,
available on virtually all Unix-like systems and Windows. The function takes
three sets of file descriptors (read, write, exception) implemented as bitmasks,
and blocks until at least one descriptor is ready or a timeout occurs. While
``select`` has limitations—notably the ``FD_SETSIZE`` limit (typically 1024)
and O(n) scanning overhead—it remains useful for applications with a moderate
number of connections and where portability is paramount.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <sys/select.h>
    #include <netinet/in.h>

    int main(void) {
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int on = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

      fd_set master, readfds;
      FD_ZERO(&master);
      FD_SET(s, &master);

      for (;;) {
        readfds = master;
        select(FD_SETSIZE, &readfds, NULL, NULL, NULL);

        for (int i = 0; i < FD_SETSIZE; i++) {
          if (!FD_ISSET(i, &readfds)) continue;
          if (i == s) {
            int c = accept(s, NULL, NULL);
            FD_SET(c, &master);
          } else {
            char buf[1024] = {0};
            ssize_t n = read(i, buf, sizeof(buf) - 1);
            if (n <= 0) { close(i); FD_CLR(i, &master); }
            else write(i, buf, n);
          }
        }
      }
    }

.. code-block:: bash

    # Terminal 1
    $ ./select-server &

    # Terminal 2
    $ nc localhost 5566
    Hello
    Hello

    # Terminal 3 (concurrent)
    $ nc localhost 5566
    World
    World

I/O Multiplexing with ``poll``
------------------------------

:Source: `src/socket/poll-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/poll-server>`_

The ``poll`` system call addresses some limitations of ``select`` by using an
array of ``pollfd`` structures instead of fixed-size bitmasks. This removes
the ``FD_SETSIZE`` limitation and allows monitoring any number of file
descriptors (limited only by system resources). Each ``pollfd`` structure
contains the file descriptor, requested events (``POLLIN`` for readable,
``POLLOUT`` for writable), and returned events filled in by the kernel.
While ``poll`` still requires O(n) scanning of all descriptors, it provides
a cleaner interface and better scalability than ``select`` for applications
with many connections. It is available on all POSIX-compliant systems.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <poll.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    #define MAX_FDS 1024

    int main(void) {
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int on = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

      struct pollfd fds[MAX_FDS];
      fds[0].fd = s;
      fds[0].events = POLLIN;
      int nfds = 1;

      for (;;) {
        poll(fds, nfds, -1);
        for (int i = 0; i < nfds; i++) {
          if (!(fds[i].revents & POLLIN)) continue;
          if (fds[i].fd == s) {
            fds[nfds].fd = accept(s, NULL, NULL);
            fds[nfds++].events = POLLIN;
          } else {
            char buf[1024] = {0};
            ssize_t n = read(fds[i].fd, buf, sizeof(buf) - 1);
            if (n <= 0) { close(fds[i].fd); fds[i] = fds[--nfds]; }
            else write(fds[i].fd, buf, n);
          }
        }
      }
    }

.. code-block:: bash

    $ ./poll-server &
    $ nc localhost 5566
    Hello poll
    Hello poll

Multithreaded TCP Server
------------------------

:Source: `src/socket/pthread-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/pthread-server>`_

For CPU-bound workloads or when simpler programming models are preferred,
spawning a dedicated thread for each client connection is a straightforward
approach. The main thread continuously accepts new connections and creates
a detached thread to handle each one independently. This model allows true
parallel execution on multi-core systems and simplifies the code since each
thread can use blocking I/O without affecting other clients. However, this
approach has scalability limits: each thread consumes memory for its stack
(typically 1-8 MB), and context switching overhead increases with many
concurrent connections. Thread-per-connection works well for servers with
moderate concurrency (hundreds of clients) or long-lived connections.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <pthread.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    void *handle(void *arg) {
      int c = *(int *)arg;
      char buf[1024];
      ssize_t n;
      while ((n = recv(c, buf, sizeof(buf), 0)) > 0)
        send(c, buf, n, 0);
      close(c);
      return NULL;
    }

    int main(void) {
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int on = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

      for (;;) {
        static int c;
        c = accept(s, NULL, NULL);
        pthread_t t;
        pthread_create(&t, NULL, handle, &c);
        pthread_detach(t);
      }
    }

.. code-block:: bash

    $ ./pthread-server &
    $ nc localhost 5566
    Hello
    Hello

I/O Multiplexing with ``epoll``/``kqueue``
------------------------------------------

:Source: `src/socket/epoll-kqueue-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/epoll-kqueue-server>`_

For high-performance servers handling thousands of concurrent connections,
Linux provides ``epoll`` and BSD/macOS provides ``kqueue``. Unlike ``select``
and ``poll`` which require scanning all file descriptors on each call, these
mechanisms use an event-driven model with O(1) complexity for adding/removing
descriptors and returning only the ready ones. With ``epoll``, you create an
epoll instance with ``epoll_create1()``, register interest in file descriptors
using ``epoll_ctl()``, and wait for events with ``epoll_wait()``. Similarly,
``kqueue`` uses ``kqueue()`` to create the queue and ``kevent()`` for both
registration and waiting. This example uses preprocessor directives to compile
on both Linux (epoll) and macOS/BSD (kqueue), demonstrating portable
high-performance I/O multiplexing.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    #ifdef __linux__
    #include <sys/epoll.h>
    #define MAX_EVENTS 64
    #elif defined(__APPLE__) || defined(__FreeBSD__)
    #include <sys/event.h>
    #define MAX_EVENTS 64
    #endif

    int main(void) {
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int on = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

      struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5566),
        .sin_addr.s_addr = INADDR_ANY
      };
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

    #ifdef __linux__
      int eq = epoll_create1(0);
      struct epoll_event ev = {.events = EPOLLIN, .data.fd = s};
      epoll_ctl(eq, EPOLL_CTL_ADD, s, &ev);
      struct epoll_event events[MAX_EVENTS];

      for (;;) {
        int n = epoll_wait(eq, events, MAX_EVENTS, -1);
        for (int i = 0; i < n; i++) {
          int fd = events[i].data.fd;
          if (fd == s) {
            int c = accept(s, NULL, NULL);
            ev.events = EPOLLIN;
            ev.data.fd = c;
            epoll_ctl(eq, EPOLL_CTL_ADD, c, &ev);
          } else {
            char buf[1024] = {0};
            ssize_t len = read(fd, buf, sizeof(buf) - 1);
            if (len <= 0) { close(fd); }
            else write(fd, buf, len);
          }
        }
      }
    #elif defined(__APPLE__) || defined(__FreeBSD__)
      int kq = kqueue();
      struct kevent ev;
      EV_SET(&ev, s, EVFILT_READ, EV_ADD, 0, 0, NULL);
      kevent(kq, &ev, 1, NULL, 0, NULL);
      struct kevent events[MAX_EVENTS];

      for (;;) {
        int n = kevent(kq, NULL, 0, events, MAX_EVENTS, NULL);
        for (int i = 0; i < n; i++) {
          int fd = (int)events[i].ident;
          if (fd == s) {
            int c = accept(s, NULL, NULL);
            EV_SET(&ev, c, EVFILT_READ, EV_ADD, 0, 0, NULL);
            kevent(kq, &ev, 1, NULL, 0, NULL);
          } else {
            char buf[1024] = {0};
            ssize_t len = read(fd, buf, sizeof(buf) - 1);
            if (len <= 0) { close(fd); }
            else write(fd, buf, len);
          }
        }
      }
    #endif
    }

.. code-block:: bash

    # Linux
    $ ./epoll-kqueue-server &
    $ nc localhost 5566
    Hello epoll
    Hello epoll

    # macOS/BSD
    $ ./epoll-kqueue-server &
    $ nc localhost 5566
    Hello kqueue
    Hello kqueue

Unix Domain Socket
------------------

:Source: `src/socket/unix-socket <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/unix-socket>`_

Unix domain sockets (also called local sockets or IPC sockets) provide
inter-process communication between processes running on the same host.
They use the filesystem namespace for addressing—the socket is represented
as a special file that must be removed before binding. Unix domain sockets
are significantly faster than TCP/IP loopback (localhost) connections because
they bypass the entire network stack: no routing, no checksums, no packet
fragmentation. They also support passing file descriptors and credentials
between processes, enabling powerful IPC patterns. Use ``AF_UNIX`` instead
of ``AF_INET`` and ``sockaddr_un`` instead of ``sockaddr_in`` for addressing.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <sys/un.h>

    int main(void) {
      unlink("/tmp/echo.sock");
      int s = socket(AF_UNIX, SOCK_STREAM, 0);

      struct sockaddr_un addr = {.sun_family = AF_UNIX};
      strncpy(addr.sun_path, "/tmp/echo.sock", sizeof(addr.sun_path) - 1);
      bind(s, (struct sockaddr *)&addr, sizeof(addr));
      listen(s, 10);

      for (;;) {
        int c = accept(s, NULL, NULL);
        char buf[1024] = {0};
        ssize_t n = recv(c, buf, sizeof(buf) - 1, 0);
        if (n > 0) send(c, buf, n, 0);
        close(c);
      }
    }

.. code-block:: bash

    $ ./unix-socket &
    $ nc -U /tmp/echo.sock
    Hello Unix
    Hello Unix

Modern DNS with ``getaddrinfo``
-------------------------------

:Source: `src/socket/getaddrinfo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/getaddrinfo>`_

The ``getaddrinfo`` function is the modern, protocol-independent replacement
for ``gethostbyname``. It handles both IPv4 and IPv6 addresses transparently,
resolves service names to port numbers, and is fully thread-safe. The function
takes hints specifying the desired address family (``AF_INET``, ``AF_INET6``,
or ``AF_UNSPEC`` for both), socket type, and protocol. It returns a linked
list of ``addrinfo`` structures, each containing a ready-to-use ``sockaddr``
that can be passed directly to ``socket()``, ``bind()``, or ``connect()``.
Always call ``freeaddrinfo()`` to release the allocated memory. This function
is the recommended way to write network code that works seamlessly with both
IPv4 and IPv6.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <netdb.h>
    #include <arpa/inet.h>

    int main(int argc, char *argv[]) {
      struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM};
      struct addrinfo *res;

      if (getaddrinfo(argv[1], NULL, &hints, &res) != 0) return 1;

      for (struct addrinfo *p = res; p; p = p->ai_next) {
        char ip[INET6_ADDRSTRLEN];
        void *addr = p->ai_family == AF_INET
          ? (void *)&((struct sockaddr_in *)p->ai_addr)->sin_addr
          : (void *)&((struct sockaddr_in6 *)p->ai_addr)->sin6_addr;
        inet_ntop(p->ai_family, addr, ip, sizeof(ip));
        printf("%s: %s\n", p->ai_family == AF_INET ? "IPv4" : "IPv6", ip);
      }
      freeaddrinfo(res);
    }

.. code-block:: bash

    $ ./getaddrinfo www.google.com
    IPv6: 2607:f8b0:4004:800::2004
    IPv4: 142.250.80.100

IP Address Conversion
---------------------

:Source: `src/socket/inet-conv <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/inet-conv>`_

The ``inet_pton`` (presentation to network) and ``inet_ntop`` (network to
presentation) functions convert IP addresses between human-readable text
strings and binary network format. Unlike the older ``inet_aton`` and
``inet_ntoa`` functions, these support both IPv4 and IPv6 addresses through
the address family parameter. The ``inet_pton`` function returns 1 on success,
0 if the input is not a valid address in the specified family, or -1 on error.
The ``inet_ntop`` function returns a pointer to the destination buffer on
success or NULL on error. These functions are essential for parsing IP
addresses from configuration files or user input and for displaying addresses
in log messages or user interfaces.

.. code-block:: c

    #include <stdio.h>
    #include <arpa/inet.h>

    int main(void) {
      // IPv4
      struct in_addr addr4;
      inet_pton(AF_INET, "192.168.1.1", &addr4);
      char str4[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &addr4, str4, sizeof(str4));
      printf("IPv4: %s (0x%x)\n", str4, ntohl(addr4.s_addr));

      // IPv6
      struct in6_addr addr6;
      inet_pton(AF_INET6, "::1", &addr6);
      char str6[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, &addr6, str6, sizeof(str6));
      printf("IPv6: %s\n", str6);
    }

.. code-block:: bash

    $ ./inet-conv
    IPv4: 192.168.1.1 (0xc0a80101)
    IPv6: ::1

Pipe Communication
------------------

:Source: `src/socket/pipe <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/pipe>`_

Pipes provide unidirectional inter-process communication. The ``pipe()`` system
call creates a pair of file descriptors: ``fd[0]`` for reading and ``fd[1]``
for writing. Data written to the write end can be read from the read end in
FIFO order. Pipes are commonly used after ``fork()`` to establish communication
between parent and child processes. The typical pattern is for one process to
close the read end and write, while the other closes the write end and reads.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>

    int main(void) {
      int fd[2];
      pipe(fd);

      pid_t pid = fork();
      if (pid == 0) {
        close(fd[1]);
        char buf[128];
        read(fd[0], buf, sizeof(buf));
        printf("Child received: %s\n", buf);
        close(fd[0]);
      } else {
        close(fd[0]);
        const char *msg = "Hello from parent";
        write(fd[1], msg, strlen(msg) + 1);
        close(fd[1]);
      }
    }

.. code-block:: bash

    $ ./pipe
    Child received: Hello from parent

Bidirectional IPC with ``socketpair``
-------------------------------------

:Source: `src/socket/socketpair <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/socket/socketpair>`_

Unlike ``pipe()`` which is unidirectional, ``socketpair()`` creates a pair of
connected Unix domain sockets that support bidirectional communication. Each
end can both read and write, making it ideal for two-way IPC between related
processes. This is simpler than creating two pipes for full-duplex communication
and provides socket semantics including the ability to pass file descriptors
between processes using ``SCM_RIGHTS``.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <sys/wait.h>

    int main(void) {
      int fd[2];
      socketpair(AF_UNIX, SOCK_STREAM, 0, fd);

      pid_t pid = fork();
      if (pid == 0) {
        close(fd[0]);
        char buf[128];
        read(fd[1], buf, sizeof(buf));
        printf("Child got: %s\n", buf);
        write(fd[1], "Hi parent", 10);
        close(fd[1]);
      } else {
        close(fd[1]);
        write(fd[0], "Hi child", 9);
        char buf[128];
        read(fd[0], buf, sizeof(buf));
        printf("Parent got: %s\n", buf);
        close(fd[0]);
        wait(NULL);
      }
    }

.. code-block:: bash

    $ ./socketpair
    Child got: Hi child
    Parent got: Hi parent
