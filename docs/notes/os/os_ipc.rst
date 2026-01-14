===========================
Inter-Process Communication
===========================

.. meta::
   :description: POSIX IPC mechanisms including pipes, FIFOs, shared memory, message queues, semaphores, and Unix domain sockets for inter-process communication in C/C++.
   :keywords: IPC, inter-process communication, pipe, FIFO, named pipe, shared memory, mmap, shm_open, message queue, mqueue, semaphore, Unix domain socket, POSIX, System V IPC

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Inter-process communication (IPC) enables processes to exchange data and
coordinate their actions. Unix-like systems provide multiple IPC mechanisms,
each with different characteristics suited to specific use cases.

The choice of IPC mechanism depends on several factors: whether processes
are related (parent-child) or unrelated, the volume of data being exchanged,
whether message boundaries matter, and performance requirements. Pipes and
FIFOs provide simple byte streams, shared memory offers the fastest data
transfer, message queues preserve message boundaries, and Unix domain sockets
provide full-duplex communication with socket semantics.

**IPC Mechanism Comparison:**

+------------------+------------+------------+------------------+------------------+
| Mechanism        | Related?   | Persistent?| Bidirectional?   | Best For         |
+==================+============+============+==================+==================+
| Pipe             | Yes        | No         | No               | Parent-child     |
+------------------+------------+------------+------------------+------------------+
| FIFO             | No         | Yes        | No               | Unrelated procs  |
+------------------+------------+------------+------------------+------------------+
| Shared Memory    | No         | Optional   | Yes              | Large data, fast |
+------------------+------------+------------+------------------+------------------+
| Message Queue    | No         | Yes        | Yes              | Structured msgs  |
+------------------+------------+------------+------------------+------------------+
| Semaphore        | No         | Yes        | N/A              | Synchronization  |
+------------------+------------+------------+------------------+------------------+
| Unix Socket      | No         | Yes        | Yes              | Full-duplex comm |
+------------------+------------+------------+------------------+------------------+

Pipes
-----

:Source: `src/ipc/pipe <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/pipe>`_

Pipes are the simplest form of IPC, providing a unidirectional byte stream
between related processes. The ``pipe()`` system call creates a pair of file
descriptors: one for reading and one for writing. Data written to the write
end can be read from the read end, following FIFO (first-in, first-out) order.

Pipes are anonymous—they exist only as long as the processes using them are
running and have no filesystem presence. This makes them ideal for parent-child
communication where the parent creates the pipe before forking. After the fork,
both processes inherit the file descriptors, and each typically closes the end
it doesn't need to establish a clear data flow direction.

The kernel buffers pipe data (typically 64KB on Linux), allowing the writer to
continue without blocking until the buffer fills. When the buffer is full, the
writer blocks until the reader consumes some data. Reading from an empty pipe
blocks until data is available or all write ends are closed (returning EOF).

.. code-block:: cpp

    #include <unistd.h>
    #include <sys/wait.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      int pipefd[2];  // [0]=read, [1]=write
      pipe(pipefd);

      if (fork() == 0) {
        // Child: read from pipe
        close(pipefd[1]);
        char buf[100];
        ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Child received: %s\n", buf);
        close(pipefd[0]);
      } else {
        // Parent: write to pipe
        close(pipefd[0]);
        const char* msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));
        close(pipefd[1]);
        wait(NULL);
      }
    }

.. code-block:: bash

    $ ./ipc-pipe
    Child received: Hello from parent!

FIFOs (Named Pipes)
-------------------

:Source: `src/ipc/fifo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/fifo>`_

FIFOs (also called named pipes) extend the pipe concept to unrelated processes
by giving the pipe a name in the filesystem. The ``mkfifo()`` function creates
a special file that acts as a rendezvous point—any process that opens this file
for reading or writing connects to the same underlying pipe.

Unlike regular pipes, FIFOs persist in the filesystem until explicitly removed
with ``unlink()``. This allows processes started at different times to
communicate. Opening a FIFO for reading blocks until another process opens it
for writing (and vice versa), unless ``O_NONBLOCK`` is specified.

FIFOs maintain the same semantics as pipes: unidirectional data flow, FIFO
ordering, and blocking behavior when the buffer is full or empty. For
bidirectional communication, you need two FIFOs or should consider Unix
domain sockets instead.

.. code-block:: cpp

    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/wait.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      const char* fifo_path = "/tmp/myfifo";
      mkfifo(fifo_path, 0666);

      if (fork() == 0) {
        // Child: write to FIFO
        int fd = open(fifo_path, O_WRONLY);
        const char* msg = "Hello via FIFO!";
        write(fd, msg, strlen(msg));
        close(fd);
      } else {
        // Parent: read from FIFO
        int fd = open(fifo_path, O_RDONLY);
        char buf[100];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Received: %s\n", buf);
        close(fd);
        wait(NULL);
        unlink(fifo_path);
      }
    }

.. code-block:: bash

    $ ./ipc-fifo
    Received: Hello via FIFO!

POSIX Shared Memory
-------------------

:Source: `src/ipc/posix-shm <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/posix-shm>`_

Shared memory is the fastest IPC mechanism because data is not copied between
processes—both processes access the same physical memory pages. POSIX shared
memory uses ``shm_open()`` to create or open a shared memory object (which
appears in ``/dev/shm`` on Linux), ``ftruncate()`` to set its size, and
``mmap()`` to map it into the process's address space.

The key advantage of shared memory is performance: once mapped, accessing
shared data is as fast as accessing regular memory. However, this speed comes
with responsibility—you must synchronize access using mutexes, semaphores, or
atomic operations to prevent data races. The kernel provides no automatic
synchronization for shared memory.

Shared memory objects persist until explicitly removed with ``shm_unlink()``,
even after all processes unmap them. This allows processes to attach and
detach at different times. The ``MAP_SHARED`` flag ensures that modifications
are visible to all processes mapping the same object.

.. code-block:: cpp

    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/wait.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      const char* shm_name = "/myshm";
      const size_t SIZE = 4096;

      // Create shared memory
      int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
      ftruncate(fd, SIZE);
      void* ptr = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);

      if (fork() == 0) {
        // Child: write to shared memory
        sprintf((char*)ptr, "Hello from child process!");
      } else {
        // Parent: wait and read
        wait(NULL);
        printf("Parent read: %s\n", (char*)ptr);
        munmap(ptr, SIZE);
        shm_unlink(shm_name);
      }
    }

.. code-block:: bash

    $ ./ipc-posix-shm
    Parent read: Hello from child process!

**Shared Memory with Synchronization:**

When multiple processes read and write shared memory concurrently, you need
synchronization. Process-shared mutexes (initialized with
``PTHREAD_PROCESS_SHARED``) can be placed in the shared memory region itself.

.. code-block:: cpp

    #include <pthread.h>

    struct SharedData {
      pthread_mutex_t mutex;
      int counter;
    };

    // Initialize (once, by creator)
    SharedData* data = (SharedData*)ptr;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&data->mutex, &attr);
    data->counter = 0;

    // Use (any process)
    pthread_mutex_lock(&data->mutex);
    data->counter++;
    pthread_mutex_unlock(&data->mutex);

POSIX Message Queues
--------------------

:Source: `src/ipc/posix-mqueue <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/posix-mqueue>`_

Message queues provide a way to exchange discrete messages between processes.
Unlike pipes which provide a byte stream, message queues preserve message
boundaries—each ``mq_send()`` creates a distinct message that is retrieved
atomically by a single ``mq_receive()``. Messages can also have priorities,
with higher-priority messages delivered first.

POSIX message queues are created with ``mq_open()``, which takes attributes
specifying the maximum number of messages and maximum message size. The queue
persists in the kernel until removed with ``mq_unlink()``. On Linux, message
queues appear as files under ``/dev/mqueue``.

Message queues are ideal when you need to send structured data units and want
the kernel to handle buffering and ordering. They're commonly used for
command/response protocols and work distribution systems. Note that POSIX
message queues are not available on macOS—use System V message queues or
other mechanisms instead.

.. code-block:: cpp

    // Note: POSIX message queues not available on macOS
    #include <mqueue.h>
    #include <fcntl.h>
    #include <sys/wait.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      const char* mq_name = "/myqueue";
      struct mq_attr attr = {.mq_maxmsg = 10, .mq_msgsize = 256};

      mqd_t mq = mq_open(mq_name, O_CREAT | O_RDWR, 0666, &attr);

      if (fork() == 0) {
        // Child: send message
        const char* msg = "Hello via message queue!";
        mq_send(mq, msg, strlen(msg) + 1, 1);  // priority = 1
        mq_close(mq);
      } else {
        // Parent: receive message
        wait(NULL);
        char buf[256];
        unsigned int prio;
        mq_receive(mq, buf, sizeof(buf), &prio);
        printf("Received (prio=%u): %s\n", prio, buf);
        mq_close(mq);
        mq_unlink(mq_name);
      }
    }

Compile with ``-lrt`` on Linux.

POSIX Semaphores
----------------

:Source: `src/ipc/posix-semaphore <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/posix-semaphore>`_

Semaphores are synchronization primitives used to coordinate access to shared
resources or signal events between processes. A semaphore maintains an integer
count: ``sem_wait()`` decrements the count (blocking if it would go negative),
and ``sem_post()`` increments it (potentially waking a blocked waiter).

POSIX provides two types of semaphores: named semaphores (created with
``sem_open()``) for unrelated processes, and unnamed semaphores (initialized
with ``sem_init()``) typically placed in shared memory. Named semaphores
persist in the filesystem (under ``/dev/shm`` on Linux) until removed with
``sem_unlink()``.

Common use cases include mutual exclusion (binary semaphore initialized to 1),
signaling between processes (initialized to 0, one process posts, another
waits), and resource counting (initialized to the number of available
resources).

.. code-block:: cpp

    #include <semaphore.h>
    #include <fcntl.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <cstdio>

    int main() {
      const char* sem_name = "/mysem";

      // Create named semaphore (initial value = 0)
      sem_t* sem = sem_open(sem_name, O_CREAT, 0666, 0);

      if (fork() == 0) {
        // Child: do work then signal
        printf("Child: doing work...\n");
        sleep(1);
        printf("Child: signaling parent\n");
        sem_post(sem);
        sem_close(sem);
      } else {
        // Parent: wait for child's signal
        printf("Parent: waiting for signal...\n");
        sem_wait(sem);
        printf("Parent: received signal!\n");
        wait(NULL);
        sem_close(sem);
        sem_unlink(sem_name);
      }
    }

.. code-block:: bash

    $ ./ipc-posix-semaphore
    Parent: waiting for signal...
    Child: doing work...
    Child: signaling parent
    Parent: received signal!

Unix Domain Sockets
-------------------

:Source: `src/ipc/unix-socket <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/unix-socket>`_

Unix domain sockets provide the full socket API for local IPC, supporting both
stream (``SOCK_STREAM``) and datagram (``SOCK_DGRAM``) modes. Unlike network
sockets, Unix domain sockets use filesystem paths for addressing and never
leave the kernel, making them faster than TCP/IP loopback.

Stream-mode Unix sockets work like TCP: a server binds to a path, listens for
connections, and accepts clients. Each accepted connection creates a new
bidirectional socket for that client. Datagram-mode sockets work like UDP,
preserving message boundaries without connection establishment.

A unique feature of Unix domain sockets is the ability to pass file descriptors
between processes using ``sendmsg()`` with ``SCM_RIGHTS`` ancillary data. This
allows one process to open a file and pass the open descriptor to another
process, enabling sophisticated privilege separation patterns.

.. code-block:: cpp

    #include <sys/socket.h>
    #include <sys/un.h>
    #include <unistd.h>
    #include <sys/wait.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      const char* sock_path = "/tmp/mysock";
      unlink(sock_path);

      if (fork() == 0) {
        // Child: client
        sleep(1);  // Wait for server
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {.sun_family = AF_UNIX};
        strcpy(addr.sun_path, sock_path);
        connect(fd, (struct sockaddr*)&addr, sizeof(addr));

        const char* msg = "Hello from client!";
        write(fd, msg, strlen(msg));

        char buf[100];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Client received: %s\n", buf);
        close(fd);
      } else {
        // Parent: server
        int srv = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {.sun_family = AF_UNIX};
        strcpy(addr.sun_path, sock_path);
        bind(srv, (struct sockaddr*)&addr, sizeof(addr));
        listen(srv, 1);

        int client = accept(srv, NULL, NULL);
        char buf[100];
        ssize_t n = read(client, buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Server received: %s\n", buf);

        const char* reply = "Hello from server!";
        write(client, reply, strlen(reply));

        close(client);
        close(srv);
        wait(NULL);
        unlink(sock_path);
      }
    }

.. code-block:: bash

    $ ./ipc-unix-socket
    Server received: Hello from client!
    Client received: Hello from server!

**Passing File Descriptors:**

Unix domain sockets can transfer open file descriptors between processes using
ancillary data. This is useful for privilege separation—a privileged process
can open restricted files and pass the descriptors to unprivileged workers.

.. code-block:: cpp

    #include <sys/socket.h>

    void send_fd(int sock, int fd) {
      char buf[1] = {0};
      struct iovec iov = {.iov_base = buf, .iov_len = 1};
      char cmsg_buf[CMSG_SPACE(sizeof(int))];
      struct msghdr msg = {
        .msg_iov = &iov, .msg_iovlen = 1,
        .msg_control = cmsg_buf, .msg_controllen = sizeof(cmsg_buf)
      };
      struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(int));
      *(int*)CMSG_DATA(cmsg) = fd;
      sendmsg(sock, &msg, 0);
    }

    int recv_fd(int sock) {
      char buf[1];
      struct iovec iov = {.iov_base = buf, .iov_len = 1};
      char cmsg_buf[CMSG_SPACE(sizeof(int))];
      struct msghdr msg = {
        .msg_iov = &iov, .msg_iovlen = 1,
        .msg_control = cmsg_buf, .msg_controllen = sizeof(cmsg_buf)
      };
      recvmsg(sock, &msg, 0);
      struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
      return *(int*)CMSG_DATA(cmsg);
    }

System V IPC (Legacy)
---------------------

:Source: `src/ipc/sysv-shm <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/ipc/sysv-shm>`_

System V IPC is an older API that predates POSIX IPC. It uses integer keys
(generated with ``ftok()``) instead of filesystem paths to identify IPC
objects. While POSIX IPC is generally preferred for new code, System V IPC
remains widely used and is available on all Unix-like systems.

System V provides three IPC mechanisms: shared memory (``shmget/shmat``),
semaphores (``semget/semop``), and message queues (``msgget/msgsnd/msgrcv``).
These objects persist in the kernel until explicitly removed and can be
inspected with the ``ipcs`` command and removed with ``ipcrm``.

.. code-block:: cpp

    #include <sys/ipc.h>
    #include <sys/shm.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <cstdio>
    #include <cstring>

    int main() {
      key_t key = ftok("/tmp", 'A');
      int shmid = shmget(key, 4096, IPC_CREAT | 0666);
      char* ptr = (char*)shmat(shmid, NULL, 0);

      if (fork() == 0) {
        // Child: write to shared memory
        strcpy(ptr, "Hello from child (System V)!");
        shmdt(ptr);
      } else {
        // Parent: wait and read
        wait(NULL);
        printf("Parent read: %s\n", ptr);
        shmdt(ptr);
        shmctl(shmid, IPC_RMID, NULL);
      }
    }

.. code-block:: bash

    $ ./ipc-sysv-shm
    Parent read: Hello from child (System V)!

**System V IPC Commands:**

.. code-block:: bash

    # List all IPC objects
    $ ipcs

    # List shared memory segments
    $ ipcs -m

    # List semaphores
    $ ipcs -s

    # List message queues
    $ ipcs -q

    # Remove shared memory segment by ID
    $ ipcrm -m <shmid>

    # Remove semaphore by ID
    $ ipcrm -s <semid>

Comparison Summary
------------------

+------------------+------------+------------+------------+------------------+
| Feature          | Pipe/FIFO  | Shm        | Msg Queue  | Unix Socket      |
+==================+============+============+============+==================+
| Speed            | Medium     | Fastest    | Medium     | Medium           |
+------------------+------------+------------+------------+------------------+
| Sync needed      | No         | Yes        | No         | No               |
+------------------+------------+------------+------------+------------------+
| Message boundary | No         | N/A        | Yes        | Datagram only    |
+------------------+------------+------------+------------+------------------+
| Bidirectional    | No         | Yes        | Yes        | Yes              |
+------------------+------------+------------+------------+------------------+
| Pass FDs         | No         | No         | No         | Yes              |
+------------------+------------+------------+------------+------------------+
