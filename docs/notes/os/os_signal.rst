======
Signal
======

.. meta::
   :description: C signal handling tutorial with examples for signal handlers, sigaction, pthread signals, blocking/unblocking signals, and child process monitoring on Unix/Linux systems.
   :keywords: C signal handling, sigaction example, signal handler, SIGINT SIGTERM, pthread signal, sigprocmask, Unix signals, Linux signal programming, POSIX signals, sigwait

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Signals are software interrupts that provide a mechanism for handling
asynchronous events in Unix-like operating systems. When a signal is
delivered to a process, the operating system interrupts the process's
normal execution flow and invokes a signal handler function.

Signals can originate from various sources: the kernel (e.g., ``SIGSEGV``
for memory access violations), other processes (via ``kill()``), terminal
input (e.g., Ctrl+C generates ``SIGINT``), or the process itself (via
``raise()`` or ``abort()``). Each signal has a default action—typically
terminating the process, ignoring the signal, or stopping/continuing
execution—but most signals can be caught and handled by user-defined
functions.

Understanding signal handling is essential for writing robust Unix
applications, particularly servers and daemons that must handle
termination requests gracefully, manage child processes, and respond
to timer events.

Print Signal Names
------------------

:Source: `src/signal/print-signals <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/print-signals>`_

The ``sys_siglist`` array (or ``strsignal()`` function) provides human-readable
names for signal numbers. This is useful for logging and debugging signal-related
issues. Note that ``sys_siglist`` is deprecated on some systems; prefer
``strsignal()`` for portable code.

.. code-block:: c

    #include <stdio.h>
    #include <signal.h>

    static int signals[] = {
        SIGABRT, SIGALRM, SIGBUS,  SIGCHLD, SIGCONT, SIGFPE,
        SIGHUP,  SIGILL,  SIGINT,  SIGIO,   SIGKILL, SIGPIPE,
        SIGPROF, SIGQUIT, SIGSEGV, SIGSYS,  SIGTERM, SIGTRAP,
        SIGTSTP, SIGTTIN, SIGTTOU, SIGURG,  SIGVTALRM, SIGUSR1,
        SIGUSR2, SIGXCPU, SIGXFSZ
    };

    int main(void) {
      for (size_t i = 0; i < sizeof(signals)/sizeof(signals[0]); i++)
        printf("Signal[%2d]: %s\n", signals[i], sys_siglist[signals[i]]);
    }

.. code-block:: console

    $ ./print-signals
    Signal[ 6]: Abort trap
    Signal[14]: Alarm clock
    Signal[10]: Bus error
    Signal[20]: Child exited
    ...

Basic Signal Handler
--------------------

:Source: `src/signal/basic-handler <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/basic-handler>`_

The ``signal()`` function registers a handler for a specific signal. When
that signal is delivered, the kernel interrupts the process and calls the
handler function with the signal number as its argument. Use ``SIG_IGN``
to ignore a signal or ``SIG_DFL`` to restore the default behavior. Note
that ``signal()`` has portability issues across Unix variants; ``sigaction()``
is preferred for new code.

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <signal.h>
    #include <unistd.h>

    void handler(int signo) {
      printf("[%d] Got signal: %s\n", getpid(), strsignal(signo));
    }

    int main(void) {
      signal(SIGHUP, handler);
      signal(SIGINT, handler);
      signal(SIGALRM, handler);
      signal(SIGUSR1, SIG_IGN);  // ignore SIGUSR1
      while (1) sleep(3);
    }

.. code-block:: console

    $ ./basic-handler &
    [54652] 
    $ kill -HUP %1
    [54652] Got signal: Hangup
    $ kill -INT %1
    [54652] Got signal: Interrupt

Signal Handling with ``sigaction``
----------------------------------

:Source: `src/signal/sigaction-handler <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/sigaction-handler>`_

The ``sigaction()`` function provides more control over signal handling than
``signal()``. It allows specifying a signal mask to block other signals during
handler execution, flags to modify behavior (e.g., ``SA_RESTART`` to restart
interrupted system calls), and access to additional signal information via
``SA_SIGINFO``. Always prefer ``sigaction()`` over ``signal()`` for reliable,
portable signal handling.

.. code-block:: c

    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>

    void handler(int signo) {
      printf("Got signal: %s\n", sys_siglist[signo]);
    }

    int main(void) {
      struct sigaction sa = {0};
      sa.sa_handler = handler;
      sigemptyset(&sa.sa_mask);
      sa.sa_flags = 0;

      sigaction(SIGINT, NULL, &sa);  // get current
      if (sa.sa_handler != SIG_IGN)
        sigaction(SIGINT, &sa, NULL);

      sigaction(SIGHUP, NULL, &sa);
      if (sa.sa_handler != SIG_IGN)
        sigaction(SIGHUP, &sa, NULL);

      printf("PID: %d\n", getpid());
      while (1) sleep(3);
    }

.. code-block:: console

    $ ./sigaction-handler &
    PID: 57140
    $ kill -HUP 57140
    Got signal: Hangup
    $ kill -INT 57140
    Got signal: Interrupt

Pthread Signal Handling
-----------------------

:Source: `src/signal/pthread-signal <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/pthread-signal>`_

In multithreaded programs, signals can be delivered to any thread that hasn't
blocked them. A common pattern is to block signals in all threads except a
dedicated signal-handling thread that uses ``sigwait()`` to synchronously
receive signals. This avoids the complexity and restrictions of asynchronous
signal handlers in threaded code.

.. code-block:: c

    #include <stdio.h>
    #include <pthread.h>
    #include <signal.h>

    void *sig_thread(void *arg) {
      sigset_t *set = (sigset_t *)arg;
      int signo;
      for (;;) {
        sigwait(set, &signo);
        printf("Got signal[%d]: %s\n", signo, sys_siglist[signo]);
      }
    }

    int main(void) {
      sigset_t set;
      sigemptyset(&set);
      sigaddset(&set, SIGQUIT);
      sigaddset(&set, SIGUSR1);
      pthread_sigmask(SIG_BLOCK, &set, NULL);

      pthread_t t;
      pthread_create(&t, NULL, sig_thread, &set);
      pause();
    }

.. code-block:: console

    $ ./pthread-signal &
    [1] 21258
    $ kill -USR1 %1
    Got signal[10]: User defined signal 1
    $ kill -QUIT %1
    Got signal[3]: Quit

Monitoring Child Processes
--------------------------

:Source: `src/signal/child-monitor <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/child-monitor>`_

When a child process terminates, the kernel sends ``SIGCHLD`` to the parent.
By installing a handler for ``SIGCHLD``, the parent can be notified immediately
when children exit, rather than blocking in ``wait()``. This is essential for
servers that spawn worker processes and need to track their lifecycle without
blocking the main event loop.

.. code-block:: c

    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>

    void handler(int signo) {
      printf("[%d] Got signal[%d]: %s\n", getpid(), signo, sys_siglist[signo]);
    }

    int main(void) {
      signal(SIGCHLD, handler);
      pid_t pid = fork();

      if (pid == 0) {
        printf("Child[%d]\n", getpid());
        sleep(3);
      } else {
        printf("Parent[%d]\n", getpid());
        pause();
      }
    }

.. code-block:: console

    $ ./child-monitor
    Parent[59113]
    Child[59114]
    [59113] Got signal[20]: Child exited

Blocking and Unblocking Signals
-------------------------------

:Source: `src/signal/signal-mask <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/signal/signal-mask>`_

The ``sigprocmask()`` function controls which signals are blocked (deferred)
for the calling thread. Blocked signals remain pending until unblocked, at
which point they are delivered. This is useful for protecting critical sections
from interruption or for temporarily deferring signal handling. Combined with
``siglongjmp()``, it enables complex control flow in signal handlers.

.. code-block:: c

    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>
    #include <setjmp.h>

    static sigjmp_buf jmpbuf;

    void handler(int signo) {
      printf("Got signal[%d]: %s\n", signo, sys_siglist[signo]);
      if (signo == SIGUSR1)
        siglongjmp(jmpbuf, 1);
    }

    int main(void) {
      sigset_t mask;
      sigemptyset(&mask);
      sigaddset(&mask, SIGHUP);

      signal(SIGHUP, handler);
      signal(SIGALRM, handler);
      signal(SIGUSR1, handler);

      if (sigsetjmp(jmpbuf, 1))
        sigprocmask(SIG_UNBLOCK, &mask, NULL);  // unblock after jump
      else
        sigprocmask(SIG_BLOCK, &mask, NULL);    // block SIGHUP

      while (1) sleep(3);
    }

.. code-block:: console

    $ ./signal-mask &
    $ kill -HUP %1      # blocked, no output
    $ kill -ALRM %1
    Got signal[14]: Alarm clock
    $ kill -USR1 %1     # triggers jump, unblocks SIGHUP
    Got signal[10]: User defined signal 1
    Got signal[1]: Hangup
