================
Thread Reference
================

.. meta::
   :description: POSIX threads (pthreads) tutorial in C with examples for thread creation, mutex locks, condition variables, read-write locks, thread-local storage, and Unix daemon programming.
   :keywords: pthread tutorial, C threads, pthread_create, pthread_join, pthread_mutex, mutex lock, condition variable, pthread_cond_wait, pthread_cond_signal, read-write lock, pthread_rwlock, thread-local storage, thread synchronization, race condition, deadlock, producer consumer, C daemon, Unix daemon, setsid, POSIX threads, multithreading C

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

POSIX threads (pthreads) provide a standardized API for creating and managing
threads in C programs. Unlike processes, threads share the same address space,
making inter-thread communication efficient through shared variables. However,
this shared memory model requires careful synchronization to prevent race
conditions, where multiple threads access shared data simultaneously and produce
unpredictable results.

The pthread library offers several synchronization primitives: mutexes provide
mutual exclusion to protect critical sections, condition variables enable threads
to wait for specific conditions, and read-write locks optimize scenarios with
frequent reads and infrequent writes. Understanding these primitives is essential
for building correct concurrent programs that avoid common pitfalls like deadlocks
and data races.

Creating Threads
----------------

:Source: `src/thread/thread-basic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/thread-basic>`_

The ``pthread_create()`` function spawns a new thread that begins execution at
the specified start routine. The function takes four arguments: a pointer to
store the thread identifier, optional thread attributes (NULL for defaults),
the function to execute, and an argument to pass to that function. The thread
runs concurrently with the calling thread until it returns or calls
``pthread_exit()``.

Use ``pthread_join()`` to wait for a thread to terminate and optionally retrieve
its return value. Failing to join or detach threads causes resource leaks.
Always check return values from pthread functions, as thread creation can fail
due to system resource limits or invalid attributes.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>

    void *worker(void *arg) {
      int id = *(int *)arg;
      printf("Thread %d running\n", id);
      return NULL;
    }

    int main(void) {
      pthread_t t1, t2;
      int id1 = 1, id2 = 2;

      pthread_create(&t1, NULL, worker, &id1);
      pthread_create(&t2, NULL, worker, &id2);

      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
      printf("All threads done\n");
    }

.. code-block:: console

    $ gcc -pthread -o thread-basic main.c && ./thread-basic
    Thread 1 running
    Thread 2 running
    All threads done

Mutex Synchronization
---------------------

:Source: `src/thread/mutex <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/mutex>`_

A mutex (mutual exclusion lock) ensures that only one thread can execute a
critical section at a time. When a thread calls ``pthread_mutex_lock()``, it
either acquires the lock immediately or blocks until the lock becomes available.
The thread must call ``pthread_mutex_unlock()`` to release the lock, allowing
other waiting threads to proceed.

Without mutex protection, concurrent access to shared variables causes race
conditions. For example, incrementing a counter involves reading the current
value, adding one, and writing back—three operations that can interleave
unpredictably across threads. Mutexes serialize these operations, ensuring
correct results at the cost of some performance overhead.

Initialize static mutexes with ``PTHREAD_MUTEX_INITIALIZER``. For dynamically
allocated mutexes, use ``pthread_mutex_init()`` and ``pthread_mutex_destroy()``.
Always pair lock and unlock calls, and consider using RAII patterns in C++ to
prevent forgetting to unlock.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>

    int counter = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

    void *increment(void *arg) {
      for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
      }
      return NULL;
    }

    int main(void) {
      pthread_t t1, t2;
      pthread_create(&t1, NULL, increment, NULL);
      pthread_create(&t2, NULL, increment, NULL);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
      printf("Counter: %d\n", counter);  // always 200000
    }

.. code-block:: console

    $ gcc -pthread -o mutex main.c && ./mutex
    Counter: 200000

Condition Variables
-------------------

:Source: `src/thread/condvar <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/condvar>`_

Condition variables allow threads to block until a particular condition becomes
true, enabling efficient producer-consumer patterns and other synchronization
scenarios. Unlike busy-waiting (repeatedly checking a condition in a loop),
condition variables put the thread to sleep until another thread signals that
the condition may have changed.

The ``pthread_cond_wait()`` function atomically releases the associated mutex
and blocks the calling thread. When the thread is awakened by a signal, it
automatically reacquires the mutex before returning. This atomic release-and-wait
prevents race conditions between checking the condition and going to sleep.

Always use a while loop to recheck the condition after ``pthread_cond_wait()``
returns, because spurious wakeups can occur and multiple threads may be competing
for the same condition. Use ``pthread_cond_signal()`` to wake one waiting thread
or ``pthread_cond_broadcast()`` to wake all waiters when the condition affects
multiple threads.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>

    int ready = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

    void *producer(void *arg) {
      pthread_mutex_lock(&lock);
      ready = 1;
      pthread_cond_signal(&cond);
      pthread_mutex_unlock(&lock);
      return NULL;
    }

    void *consumer(void *arg) {
      pthread_mutex_lock(&lock);
      while (!ready)
        pthread_cond_wait(&cond, &lock);
      printf("Consumer: data ready\n");
      pthread_mutex_unlock(&lock);
      return NULL;
    }

    int main(void) {
      pthread_t prod, cons;
      pthread_create(&cons, NULL, consumer, NULL);
      pthread_create(&prod, NULL, producer, NULL);
      pthread_join(prod, NULL);
      pthread_join(cons, NULL);
    }

.. code-block:: console

    $ gcc -pthread -o condvar main.c && ./condvar
    Consumer: data ready

Read-Write Locks
----------------

:Source: `src/thread/rwlock <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/rwlock>`_

Read-write locks optimize access patterns where reads are frequent and writes
are rare. Multiple threads can hold the read lock simultaneously, allowing
concurrent read access to shared data. However, the write lock is exclusive—only
one thread can hold it, and no readers are allowed while a writer has the lock.

Use ``pthread_rwlock_rdlock()`` for read access and ``pthread_rwlock_wrlock()``
for write access. This distinction improves throughput compared to a regular
mutex when most operations are reads, as readers don't block each other. However,
read-write locks have higher overhead than mutexes, so they're only beneficial
when the read-to-write ratio is high and critical sections are long enough to
amortize the locking cost.

Be aware of writer starvation: if readers continuously acquire the lock, writers
may wait indefinitely. Some implementations provide fairness policies to prevent
this, but behavior varies across platforms.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>

    int data = 0;
    pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

    void *reader(void *arg) {
      pthread_rwlock_rdlock(&rwlock);
      printf("Read: %d\n", data);
      pthread_rwlock_unlock(&rwlock);
      return NULL;
    }

    void *writer(void *arg) {
      pthread_rwlock_wrlock(&rwlock);
      data = 42;
      printf("Wrote: %d\n", data);
      pthread_rwlock_unlock(&rwlock);
      return NULL;
    }

    int main(void) {
      pthread_t r1, r2, w;
      pthread_create(&w, NULL, writer, NULL);
      pthread_create(&r1, NULL, reader, NULL);
      pthread_create(&r2, NULL, reader, NULL);
      pthread_join(w, NULL);
      pthread_join(r1, NULL);
      pthread_join(r2, NULL);
    }

.. code-block:: console

    $ gcc -pthread -o rwlock main.c && ./rwlock
    Wrote: 42
    Read: 42
    Read: 42

Thread-Local Storage
--------------------

:Source: `src/thread/thread-local <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/thread-local>`_

Thread-local storage (TLS) provides each thread with its own instance of a
variable, eliminating the need for synchronization when threads need private
copies of data. This is useful for per-thread caches, error codes, or context
that shouldn't be shared across threads.

The ``__thread`` keyword (GCC/Clang extension) or ``_Thread_local`` (C11 standard)
declares thread-local variables with static storage duration. Each thread gets
its own copy, initialized independently. For more complex scenarios requiring
dynamic allocation or destructor callbacks, use ``pthread_key_create()`` with
``pthread_setspecific()`` and ``pthread_getspecific()``.

Thread-local storage is more efficient than passing context through function
arguments when the data is needed deep in the call stack, and it avoids the
overhead of mutex locking for thread-private data.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>

    __thread int tls_var = 0;

    void *worker(void *arg) {
      tls_var = *(int *)arg;
      printf("Thread %d: tls_var = %d\n", tls_var, tls_var);
      return NULL;
    }

    int main(void) {
      pthread_t t1, t2;
      int id1 = 1, id2 = 2;
      pthread_create(&t1, NULL, worker, &id1);
      pthread_create(&t2, NULL, worker, &id2);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);
    }

.. code-block:: console

    $ gcc -pthread -o thread-local main.c && ./thread-local
    Thread 1: tls_var = 1
    Thread 2: tls_var = 2

Creating a Unix Daemon
----------------------

:Source: `src/thread/daemon <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/daemon>`_

A daemon is a background process that runs independently of any controlling
terminal, typically providing system services like web servers, database engines,
or scheduled tasks. Creating a proper daemon requires several steps to fully
detach from the parent process and terminal environment.

The traditional daemonization sequence involves: calling ``fork()`` to create a
child process (parent exits, ensuring the child isn't a process group leader),
calling ``setsid()`` to create a new session and become the session leader,
optionally forking again to prevent acquiring a controlling terminal, changing
the working directory to root to avoid blocking filesystem unmounts, resetting
the file creation mask with ``umask()``, and closing standard file descriptors
to release terminal resources.

Since stdout and stderr are closed, daemons typically use ``syslog()`` for
logging. The syslog facility provides centralized, persistent logging with
severity levels and is the standard approach for daemon diagnostics.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <syslog.h>
    #include <sys/stat.h>

    int main(void) {
      pid_t pid = fork();
      if (pid < 0) exit(1);
      if (pid > 0) exit(0);  // parent exits

      umask(0);
      if (setsid() < 0) exit(1);
      if (chdir("/") < 0) exit(1);

      close(STDIN_FILENO);
      close(STDOUT_FILENO);
      close(STDERR_FILENO);

      while (1) {
        sleep(3);
        syslog(LOG_INFO, "Daemon running");
      }
    }

.. code-block:: console

    $ ./daemon
    $ ps aux | grep daemon
    user  12345  0.0  0.0  daemon

Using the ``daemon()`` Function
-------------------------------

:Source: `src/thread/daemon-simple <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/thread/daemon-simple>`_

The BSD ``daemon()`` function provides a convenient wrapper that handles the
standard daemonization steps in a single call. The first argument ``nochdir``
controls whether to change the working directory to root (0 means change, nonzero
means stay). The second argument ``noclose`` controls whether to redirect stdin,
stdout, and stderr to /dev/null (0 means redirect, nonzero means keep open).

While ``daemon()`` simplifies the code, it's not part of the POSIX standard and
may not be available on all Unix-like systems. For maximum portability across
different platforms, implement the daemonization steps manually. Additionally,
modern systems often use systemd or other init systems that handle daemonization
externally, making the traditional double-fork approach less necessary for new
services.

.. code-block:: c

    #include <stdio.h>
    #include <unistd.h>
    #include <syslog.h>

    int main(void) {
      if (daemon(0, 0) < 0) {
        syslog(LOG_ERR, "daemon() failed");
        return 1;
      }

      while (1) {
        sleep(3);
        syslog(LOG_INFO, "Daemon running");
      }
    }

.. code-block:: console

    $ ./daemon-simple
    $ pgrep daemon-simple
    12346
