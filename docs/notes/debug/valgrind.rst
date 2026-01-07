==================
Valgrind Reference
==================

.. meta::
   :description: Valgrind tutorial for memory leak detection, invalid memory access, use-after-free, cache profiling with Cachegrind, and call graph analysis with Callgrind in C/C++ programs.
   :keywords: Valgrind tutorial, memcheck, memory leak detection, invalid memory access, use-after-free, double free, cachegrind, callgrind, helgrind, heap profiling, C++ memory debugging, Valgrind examples, uninitialized memory

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Valgrind is a dynamic binary instrumentation framework that provides powerful
tools for detecting memory errors, profiling cache behavior, and analyzing
program execution. It works by running your program in a virtual machine that
intercepts every memory access, enabling detection of bugs that would otherwise
cause silent corruption, security vulnerabilities, or intermittent crashes.

The most widely used Valgrind tool is Memcheck, which detects memory leaks,
use of uninitialized memory, buffer overflows, use-after-free errors, double
frees, and mismatched allocation/deallocation (e.g., using ``free()`` on memory
allocated with ``new``). These errors are notoriously difficult to debug because
they often don't crash immediately—instead, they corrupt memory that may not be
accessed until much later.

While Valgrind adds significant overhead (typically 10-50x slower execution),
it finds bugs that are nearly impossible to detect through code review or
testing alone. The detailed error reports include full stack traces showing
exactly where the error occurred and where the problematic memory was allocated.

Memcheck: Memory Errors
-----------------------

Memcheck is Valgrind's default tool and the most commonly used. It tracks every
byte of memory, recording whether it has been allocated, initialized, and freed.
When your program accesses memory incorrectly, Memcheck reports the error with
a stack trace.

.. code-block:: bash

    $ gcc -g -o myprogram main.c
    $ valgrind ./myprogram
    $ valgrind --leak-check=full ./myprogram
    $ valgrind --leak-check=full --show-leak-kinds=all ./myprogram

Common options:

.. code-block:: bash

    --leak-check=full          # detailed leak information
    --show-leak-kinds=all      # show all leak types
    --track-origins=yes        # track uninitialized value origins
    --verbose                  # more detailed output
    -v                         # verbose
    --log-file=valgrind.log    # write output to file

Example output for memory leak:

.. code-block:: text

    ==12345== HEAP SUMMARY:
    ==12345==     in use at exit: 100 bytes in 1 blocks
    ==12345==   total heap usage: 10 allocs, 9 frees, 1,024 bytes allocated
    ==12345==
    ==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
    ==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
    ==12345==    by 0x400537: main (main.c:10)

Example output for invalid read:

.. code-block:: text

    ==12345== Invalid read of size 4
    ==12345==    at 0x400520: main (main.c:8)
    ==12345==  Address 0x5204044 is 0 bytes after a block of size 4 alloc'd
    ==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
    ==12345==    by 0x400510: main (main.c:6)

Leak Types
----------

Valgrind categorizes memory leaks by how reachable the lost memory is from
pointers still in scope. Understanding these categories helps prioritize which
leaks to fix first.

.. code-block:: text

    definitely lost    - no pointer to block exists, memory is leaked
    indirectly lost    - pointer exists only in a "definitely lost" block
    possibly lost      - pointer to interior of block (may be intentional)
    still reachable    - pointer exists at exit, but memory not freed

"Definitely lost" blocks are the most serious—they represent memory that your
program allocated but can no longer free because all pointers to it are gone.
"Still reachable" blocks are often acceptable, especially for global data
structures that live for the program's lifetime.

Suppressing False Positives
---------------------------

Some libraries have known memory behaviors that Valgrind reports as errors but
are actually intentional. Create suppression files to silence these warnings:

.. code-block:: bash

    $ valgrind --gen-suppressions=all ./myprogram 2>&1 | grep -A 10 "{"
    $ valgrind --suppressions=myapp.supp ./myprogram

Cachegrind: Cache Profiling
---------------------------

Cachegrind simulates the CPU cache hierarchy to identify code with poor cache
utilization. Cache misses are a major source of performance problems because
accessing main memory is 100x slower than accessing L1 cache. Cachegrind counts
instruction fetches, data reads/writes, and cache misses at each level.

.. code-block:: bash

    $ valgrind --tool=cachegrind ./myprogram
    $ cg_annotate cachegrind.out.<pid>
    $ cg_annotate --auto=yes cachegrind.out.<pid>   # annotate source

Output shows cache misses:

.. code-block:: text

    I   refs:      1,000,000
    I1  misses:        1,000
    I1  miss rate:      0.1%
    D   refs:        500,000
    D1  misses:       10,000
    D1  miss rate:      2.0%
    LL  misses:        5,000
    LL  miss rate:      0.3%

Callgrind: Call Graph Profiling
-------------------------------

Callgrind records the call history of a program, counting instructions executed
in each function and tracking caller-callee relationships. Unlike sampling
profilers, Callgrind counts every instruction, providing exact measurements
rather than statistical estimates. The output can be visualized with KCachegrind,
which displays interactive call graphs and source annotations.

.. code-block:: bash

    $ valgrind --tool=callgrind ./myprogram
    $ callgrind_annotate callgrind.out.<pid>
    $ kcachegrind callgrind.out.<pid>    # GUI visualization

Options:

.. code-block:: bash

    --callgrind-out-file=callgrind.out   # custom output file
    --dump-instr=yes                      # instruction-level profiling
    --collect-jumps=yes                   # branch prediction info

Helgrind: Thread Errors
-----------------------

Helgrind detects synchronization errors in multithreaded programs, including
data races (concurrent access to shared data without proper locking), deadlocks
(circular lock dependencies), and misuse of the POSIX threads API. Data races
are particularly insidious because they may only manifest under specific timing
conditions, making them difficult to reproduce and debug.

.. code-block:: bash

    $ valgrind --tool=helgrind ./myprogram

Helgrind detects:

- Data races (concurrent access without synchronization)
- Lock order violations (potential deadlocks)
- Misuse of pthread API (destroying locked mutex, unlocking unowned mutex)

Common Workflows
----------------

**Find memory leaks:**

.. code-block:: bash

    $ valgrind --leak-check=full --show-leak-kinds=all ./myprogram

**Track uninitialized values:**

.. code-block:: bash

    $ valgrind --track-origins=yes ./myprogram

**Profile cache performance:**

.. code-block:: bash

    $ valgrind --tool=cachegrind ./myprogram
    $ cg_annotate --auto=yes cachegrind.out.*

**Generate call graph:**

.. code-block:: bash

    $ valgrind --tool=callgrind ./myprogram
    $ kcachegrind callgrind.out.*   # requires KCachegrind GUI
