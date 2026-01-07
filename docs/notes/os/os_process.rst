=================
Process Reference
=================

.. meta::
   :description: Unix process management tutorial in C with examples for fork, exec, wait, waitpid, process IDs, environment variables, and system information on Linux and macOS.
   :keywords: fork exec wait, C process tutorial, fork example, execvp execve, waitpid WNOHANG, process creation, Unix process management, getpid getppid, zombie process, orphan process, environment variables, getenv setenv, uname gethostname, process group, session leader, WIFEXITED WIFSIGNALED

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Unix processes are the fundamental unit of execution, each with its own address
space, file descriptors, and execution context. The ``fork()`` system call creates
new processes by duplicating the calling process, while the ``exec()`` family
replaces the current process image with a new program. Together, fork and exec
form the basis for running programs, building shells, and implementing job control.

Process management also involves waiting for child processes to terminate using
``wait()`` or ``waitpid()``, which prevents zombie processes (terminated children
whose exit status hasn't been collected). Understanding process relationships—
parent-child hierarchies, process groups, and sessions—is essential for building
robust system software, daemons, and interactive applications.

Fork and Wait
-------------

:Source: `src/process/fork-wait <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/fork-wait>`_

The ``fork()`` system call creates a new process by duplicating the calling
process. After fork returns, two processes continue execution from the same
point: the parent receives the child's process ID as the return value, while
the child receives zero. A negative return value indicates fork failed, typically
due to resource limits.

The ``wait()`` function blocks the parent until any child process terminates,
returning the child's PID and storing its exit status. Use the ``WEXITSTATUS()``
macro to extract the exit code from the status value. Failing to wait for
children creates zombie processes—entries in the process table that consume
system resources until the parent collects their status or terminates.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/wait.h>

    int main(void) {
      pid_t pid = fork();
      if (pid < 0) {
        perror("fork");
        exit(1);
      }
      if (pid == 0) {
        printf("Child PID: %d\n", getpid());
        exit(42);
      }
      int status;
      wait(&status);
      printf("Child exited with %d\n", WEXITSTATUS(status));
    }

.. code-block:: console

    $ ./fork-wait
    Child PID: 12345
    Child exited with 42

Fork and Exec
-------------

:Source: `src/process/fork-exec <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/fork-exec>`_

The ``exec()`` family of functions replaces the current process image with a
new program. After a successful exec call, the original code is completely
replaced—exec only returns if an error occurs. The various exec variants differ
in how they accept arguments and whether they search the PATH environment
variable.

``execvp()`` takes an argument vector (array of strings) and searches PATH for
the executable, making it convenient for shell-like behavior. ``execve()`` is
the underlying system call that requires the full path and explicit environment.
Other variants like ``execl()`` and ``execlp()`` accept arguments as variadic
parameters rather than an array.

The fork-exec pattern is fundamental to Unix: fork creates a child process,
which then calls exec to run a different program. This separation allows the
parent to set up the child's environment (redirecting file descriptors, changing
directories, setting resource limits) before the new program starts.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/wait.h>

    int main(void) {
      pid_t pid = fork();
      if (pid < 0) {
        perror("fork");
        exit(1);
      }
      if (pid == 0) {
        char *args[] = {"ls", "-l", NULL};
        execvp("ls", args);
        perror("execvp");  // only reached on error
        exit(1);
      }
      wait(NULL);
      printf("Child finished\n");
    }

.. code-block:: console

    $ ./fork-exec
    total 16
    -rwxr-xr-x  1 user  staff  8432 Jan  6 10:00 fork-exec
    Child finished

Waitpid with Options
--------------------

:Source: `src/process/waitpid <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/waitpid>`_

The ``waitpid()`` function provides more control than ``wait()``, allowing you
to wait for a specific child process or use non-blocking checks. The first
argument specifies which child to wait for: a positive PID waits for that
specific child, -1 waits for any child (like ``wait()``), 0 waits for any child
in the same process group, and a negative value waits for any child in the
specified process group.

The ``WNOHANG`` option makes waitpid return immediately if no child has exited,
returning 0 instead of blocking. This enables polling patterns where the parent
can do other work while periodically checking for child termination. The
``WUNTRACED`` option also reports children that have been stopped by a signal.

Use the status inspection macros to determine how the child terminated:
``WIFEXITED()`` checks for normal exit, ``WIFSIGNALED()`` checks if killed by
a signal, and ``WIFSTOPPED()`` checks if stopped. Extract details with
``WEXITSTATUS()``, ``WTERMSIG()``, and ``WSTOPSIG()`` respectively.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/wait.h>

    int main(void) {
      pid_t pid = fork();
      if (pid == 0) {
        sleep(1);
        exit(0);
      }

      int status;
      while (waitpid(pid, &status, WNOHANG) == 0) {
        printf("Child still running...\n");
        usleep(200000);
      }

      if (WIFEXITED(status))
        printf("Exited: %d\n", WEXITSTATUS(status));
      else if (WIFSIGNALED(status))
        printf("Killed by signal: %d\n", WTERMSIG(status));
    }

.. code-block:: console

    $ ./waitpid
    Child still running...
    Child still running...
    Child still running...
    Child still running...
    Child still running...
    Exited: 0

Process IDs
-----------

:Source: `src/process/pid <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/pid>`_

Every process has several associated identifiers that define its place in the
process hierarchy. ``getpid()`` returns the process's own ID, while ``getppid()``
returns the parent's ID. When a parent terminates before its children, those
orphaned processes are adopted by init (PID 1) or a subreaper process.

Process groups and sessions organize processes for job control. ``getpgrp()``
returns the process group ID, which groups related processes (like a pipeline)
for signal delivery. ``getsid()`` returns the session ID, which groups process
groups under a controlling terminal. These concepts are essential for implementing
shells and understanding how signals like SIGINT (Ctrl+C) are delivered to
foreground process groups.

.. code-block:: c

    #include <stdio.h>
    #include <unistd.h>

    int main(void) {
      printf("PID:  %d\n", getpid());
      printf("PPID: %d\n", getppid());
      printf("PGRP: %d\n", getpgrp());
      printf("SID:  %d\n", getsid(0));
    }

.. code-block:: console

    $ ./pid
    PID:  12345
    PPID: 12300
    PGRP: 12345
    SID:  12300

Environment Variables
---------------------

:Source: `src/process/environ <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/environ>`_

Environment variables provide a way to pass configuration to processes without
command-line arguments. Each process inherits its parent's environment, which
can be modified before calling exec to customize the child's environment.
Common variables include PATH (executable search directories), HOME (user's
home directory), and LANG (locale settings).

``getenv()`` retrieves a variable's value, returning NULL if not set.
``setenv()`` sets a variable, with the third argument controlling whether to
overwrite an existing value. ``unsetenv()`` removes a variable from the
environment. These functions modify the process's own environment; changes
don't affect the parent or other processes.

The global ``environ`` variable points to the full environment array, useful
for iterating over all variables or passing to ``execve()``. Environment
variables are commonly used for configuration that shouldn't be hardcoded,
such as API keys, database URLs, or feature flags.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    int main(void) {
      printf("HOME: %s\n", getenv("HOME"));

      setenv("MY_VAR", "hello", 1);
      printf("MY_VAR: %s\n", getenv("MY_VAR"));

      unsetenv("MY_VAR");
      printf("MY_VAR after unset: %s\n", getenv("MY_VAR"));
    }

.. code-block:: console

    $ ./environ
    HOME: /home/user
    MY_VAR: hello
    MY_VAR after unset: (null)

System Information
------------------

:Source: `src/process/sysinfo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/process/sysinfo>`_

The ``uname()`` system call fills a ``struct utsname`` with information about
the operating system and hardware. Fields include ``sysname`` (OS name like
"Linux" or "Darwin"), ``nodename`` (network hostname), ``release`` (kernel
version), ``version`` (build information), and ``machine`` (hardware architecture
like "x86_64" or "arm64").

``gethostname()`` retrieves just the system's hostname, which may differ from
the nodename in utsname on some systems. These functions are useful for logging,
generating platform-specific behavior, or displaying system information to users.
The information returned reflects the running kernel, not necessarily the
distribution or installed packages.

.. code-block:: c

    #include <stdio.h>
    #include <sys/utsname.h>
    #include <unistd.h>

    int main(void) {
      struct utsname buf;
      uname(&buf);
      printf("OS:      %s\n", buf.sysname);
      printf("Host:    %s\n", buf.nodename);
      printf("Release: %s\n", buf.release);
      printf("Arch:    %s\n", buf.machine);

      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("Hostname: %s\n", hostname);
    }

.. code-block:: console

    $ ./sysinfo
    OS:      Linux
    Host:    myhost
    Release: 5.15.0-generic
    Arch:    x86_64
    Hostname: myhost
