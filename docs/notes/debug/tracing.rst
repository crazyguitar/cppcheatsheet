=================
Tracing Reference
=================

.. meta::
   :description: Linux tracing tutorial with strace for system call tracing, ltrace for library call tracing, and ftrace for kernel function tracing in C/C++ debugging and analysis.
   :keywords: strace tutorial, ltrace, ftrace, system call tracing, library tracing, kernel tracing, Linux debugging, syscall debugging, trace process, perf trace, trace-cmd

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Tracing tools intercept and log system calls, library calls, or kernel functions
as a program executes. Unlike debuggers that pause execution at breakpoints,
tracers observe programs with minimal interference, making them ideal for
understanding program behavior, diagnosing I/O issues, debugging permission
problems, and reverse engineering unfamiliar code.

strace traces system calls—the interface between user programs and the Linux
kernel—showing every file operation, network connection, and process management
call. ltrace traces calls to shared library functions like libc, revealing how
programs use standard library APIs. ftrace traces kernel functions, useful for
driver development and understanding kernel behavior.

These tools work on any executable without recompilation or debug symbols,
making them invaluable for debugging production systems and third-party software.

strace: System Call Tracing
---------------------------

strace intercepts and records every system call made by a program, showing the
call name, arguments, and return value. This reveals how programs interact with
the operating system: which files they open, what network connections they make,
how they spawn child processes, and where they spend time waiting.

Basic usage:

.. code-block:: bash

    $ strace ./myprogram              # trace program
    $ strace -p 1234                  # attach to running process
    $ strace -f ./myprogram           # follow child processes
    $ strace -o trace.log ./myprogram # output to file

Filter by system call:

.. code-block:: bash

    $ strace -e open ./myprogram           # only open() calls
    $ strace -e trace=open,read,write ./prog  # multiple syscalls
    $ strace -e trace=file ./myprogram     # file-related syscalls
    $ strace -e trace=network ./myprogram  # network syscalls
    $ strace -e trace=process ./myprogram  # process syscalls
    $ strace -e trace=memory ./myprogram   # memory syscalls

Useful options:

.. code-block:: bash

    -c                  # count time, calls, and errors per syscall
    -T                  # show time spent in each syscall
    -t                  # timestamp each line
    -tt                 # timestamp with microseconds
    -r                  # relative timestamp (time since last syscall)
    -s 1000             # print 1000 chars of strings (default 32)
    -y                  # print paths for file descriptors
    -yy                 # print socket details

Example output:

.. code-block:: text

    open("config.txt", O_RDONLY)        = 3
    read(3, "key=value\n", 4096)        = 10
    close(3)                            = 0
    write(1, "Hello, World!\n", 14)     = 14

Count syscalls:

.. code-block:: bash

    $ strace -c ./myprogram
    % time     seconds  usecs/call     calls    errors syscall
    ------ ----------- ----------- --------- --------- ----------------
     45.00    0.000450          45        10           read
     30.00    0.000300          30        10           write
     25.00    0.000250          25        10         5 open
    ------ ----------- ----------- --------- --------- ----------------
    100.00    0.001000                    30         5 total

Common Workflows
^^^^^^^^^^^^^^^^

**Debug file not found errors:**

.. code-block:: bash

    $ strace -e open,openat ./myprogram 2>&1 | grep -i "no such file"

**See what files a program opens:**

.. code-block:: bash

    $ strace -e trace=file -y ./myprogram

**Debug network issues:**

.. code-block:: bash

    $ strace -e trace=network -s 1000 ./myprogram

**Find why a program hangs:**

.. code-block:: bash

    $ strace -p $(pgrep myprogram)
    # Shows which syscall is blocking

ltrace: Library Call Tracing
----------------------------

ltrace intercepts calls to shared library functions, showing how programs use
libc, libm, and other libraries. This is useful for understanding program
behavior at a higher level than system calls—seeing string operations, memory
allocations, and mathematical functions rather than raw kernel interfaces.

.. code-block:: bash

    $ ltrace ./myprogram              # trace library calls
    $ ltrace -p 1234                  # attach to process
    $ ltrace -e malloc+free ./prog    # specific functions
    $ ltrace -c ./myprogram           # count calls

Example output:

.. code-block:: text

    malloc(100)                       = 0x5555556
    strcpy(0x5555556, "hello")        = 0x5555556
    strlen("hello")                   = 5
    printf("Length: %d\n", 5)         = 10
    free(0x5555556)                   = <void>

Options:

.. code-block:: bash

    -e expr             # filter functions (e.g., -e malloc+free)
    -l libname          # trace only calls to specific library
    -s 1000             # string length limit
    -n 2                # indent nested calls
    -c                  # count time and calls
    -S                  # trace system calls too (like strace)

ftrace: Kernel Function Tracing
-------------------------------

ftrace is a kernel tracing framework built into Linux, accessed through the
debugfs filesystem. It traces kernel functions with very low overhead, making
it suitable for production systems. ftrace is essential for kernel and driver
development, helping understand how the kernel processes system calls and
handles hardware interrupts.

Setup:

.. code-block:: bash

    # Mount debugfs if not mounted
    $ sudo mount -t debugfs none /sys/kernel/debug

    # ftrace files are in /sys/kernel/debug/tracing/
    $ cd /sys/kernel/debug/tracing

Basic function tracing:

.. code-block:: bash

    # List available tracers
    $ cat available_tracers
    function function_graph nop

    # Enable function tracer
    $ echo function > current_tracer

    # Filter specific functions
    $ echo 'tcp_*' > set_ftrace_filter

    # Start tracing
    $ echo 1 > tracing_on

    # Run your program
    $ ./myprogram

    # Stop and view trace
    $ echo 0 > tracing_on
    $ cat trace

Function graph tracer shows the call hierarchy with timing:

.. code-block:: bash

    $ echo function_graph > current_tracer
    $ echo 1 > tracing_on
    $ ./myprogram
    $ echo 0 > tracing_on
    $ cat trace

Example output:

.. code-block:: text

     0)               |  sys_read() {
     0)               |    vfs_read() {
     0)   0.500 us    |      rw_verify_area();
     0)               |      __vfs_read() {
     0)   1.200 us    |        ext4_file_read_iter();
     0)   1.500 us    |      }
     0)   2.500 us    |    }
     0)   3.000 us    |  }

trace-cmd (ftrace frontend):

.. code-block:: bash

    $ sudo trace-cmd record -p function -l 'tcp_*' ./myprogram
    $ trace-cmd report

perf trace
----------

perf also provides system call tracing with lower overhead than strace, making
it more suitable for tracing performance-sensitive applications or production
systems where strace's overhead would be problematic:

.. code-block:: bash

    $ perf trace ./myprogram          # trace syscalls
    $ perf trace -p 1234              # attach to process
    $ perf trace -e open ./myprogram  # specific syscall
    $ perf trace -s ./myprogram       # summary statistics

Comparison
----------

.. code-block:: text

    Tool        Traces              Overhead    Use Case
    ─────────────────────────────────────────────────────────────────
    strace      System calls        High        Debug I/O, files, network
    ltrace      Library calls       Medium      Debug libc usage
    ftrace      Kernel functions    Low         Kernel/driver debugging
    perf trace  System calls        Low         Production tracing
