====================
Sanitizers Reference
====================

.. meta::
   :description: Compiler sanitizers tutorial for AddressSanitizer (ASan), ThreadSanitizer (TSan), UndefinedBehaviorSanitizer (UBSan), MemorySanitizer (MSan), and LeakSanitizer for C/C++ debugging.
   :keywords: AddressSanitizer, ASan, ThreadSanitizer, TSan, UndefinedBehaviorSanitizer, UBSan, MemorySanitizer, MSan, LeakSanitizer, memory debugging, data race detection, undefined behavior, buffer overflow, use-after-free

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Sanitizers are compile-time instrumentation tools built into GCC and Clang that
detect bugs at runtime with significantly lower overhead than Valgrind—typically
2-5x slowdown compared to Valgrind's 10-50x. They work by inserting runtime
checks around memory accesses, thread operations, or arithmetic operations
during compilation, catching errors as they occur.

Each sanitizer targets specific bug classes: AddressSanitizer catches memory
errors like buffer overflows and use-after-free, ThreadSanitizer detects data
races in multithreaded code, UndefinedBehaviorSanitizer catches undefined
behavior like integer overflow and null pointer dereference, and MemorySanitizer
finds reads of uninitialized memory.

Because sanitizers are integrated into the compiler, they understand your code's
structure and can provide detailed error reports with source locations. They're
designed to be used during development and testing, catching bugs before they
reach production. Many organizations run sanitizer-enabled builds in their
continuous integration pipelines.

AddressSanitizer (ASan)
-----------------------

AddressSanitizer is the most widely used sanitizer, detecting memory errors that
cause security vulnerabilities and crashes. It catches heap buffer overflows,
stack buffer overflows, use-after-free, use-after-return, use-after-scope,
double-free, and memory leaks. ASan uses shadow memory to track the state of
every byte of application memory, enabling fast checks on every memory access.

.. code-block:: bash

    $ gcc -fsanitize=address -g -o myprogram main.c
    $ ./myprogram

    $ clang -fsanitize=address -g -o myprogram main.c
    $ ./myprogram

Example output for heap buffer overflow:

.. code-block:: text

    ==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014
    READ of size 4 at 0x602000000014 thread T0
        #0 0x4005a7 in main main.c:8
        #1 0x7f1234 in __libc_start_main

    0x602000000014 is located 0 bytes to the right of 20-byte region
    allocated by thread T0 here:
        #0 0x7f5678 in malloc
        #1 0x400567 in main main.c:5

Environment variables:

.. code-block:: bash

    ASAN_OPTIONS=detect_leaks=1          # enable leak detection (default on Linux)
    ASAN_OPTIONS=abort_on_error=1        # abort instead of exit
    ASAN_OPTIONS=symbolize=1             # symbolize stack traces
    ASAN_OPTIONS=halt_on_error=0         # continue after first error

LeakSanitizer (LSan)
--------------------

LeakSanitizer detects memory leaks by scanning memory at program exit to find
allocated blocks that are no longer reachable. It's included with ASan by
default on Linux, providing leak detection with minimal additional overhead.
LSan can also be used standalone when you only need leak detection without
the full memory error checking of ASan.

.. code-block:: bash

    $ gcc -fsanitize=leak -g -o myprogram main.c
    $ ./myprogram

    # Or with ASan (leak detection enabled by default)
    $ gcc -fsanitize=address -g -o myprogram main.c
    $ ASAN_OPTIONS=detect_leaks=1 ./myprogram

Example output:

.. code-block:: text

    ==12345==ERROR: LeakSanitizer: detected memory leaks

    Direct leak of 100 bytes in 1 object(s) allocated from:
        #0 0x7f1234 in malloc
        #1 0x400537 in main main.c:10

    SUMMARY: LeakSanitizer: 100 byte(s) leaked in 1 allocation(s).

ThreadSanitizer (TSan)
----------------------

ThreadSanitizer detects data races, which occur when two threads access the
same memory location concurrently, at least one access is a write, and there's
no synchronization between them. Data races cause undefined behavior and lead
to bugs that are extremely difficult to reproduce because they depend on thread
scheduling. TSan also detects deadlocks and improper use of synchronization
primitives.

.. code-block:: bash

    $ gcc -fsanitize=thread -g -o myprogram main.c -pthread
    $ ./myprogram

    $ clang -fsanitize=thread -g -o myprogram main.c -pthread
    $ ./myprogram

Example output for data race:

.. code-block:: text

    WARNING: ThreadSanitizer: data race (pid=12345)
      Write of size 4 at 0x7f1234 by thread T1:
        #0 increment main.c:10

      Previous read of size 4 at 0x7f1234 by thread T2:
        #0 increment main.c:10

      Location is global 'counter' of size 4 at 0x7f1234

Environment variables:

.. code-block:: bash

    TSAN_OPTIONS=halt_on_error=1         # stop on first error
    TSAN_OPTIONS=second_deadlock_stack=1 # show both stacks for deadlock

UndefinedBehaviorSanitizer (UBSan)
----------------------------------

UndefinedBehaviorSanitizer detects operations that the C/C++ standards declare
as undefined behavior. When undefined behavior occurs, the compiler is free to
do anything—including appearing to work correctly until a different compiler
version or optimization level causes failures. UBSan catches these issues at
runtime, including signed integer overflow, null pointer dereference, misaligned
pointer access, out-of-bounds array indexing, and invalid shift operations.

.. code-block:: bash

    $ gcc -fsanitize=undefined -g -o myprogram main.c
    $ ./myprogram

    $ clang -fsanitize=undefined -g -o myprogram main.c
    $ ./myprogram

Specific checks:

.. code-block:: bash

    -fsanitize=signed-integer-overflow   # signed overflow
    -fsanitize=unsigned-integer-overflow # unsigned overflow (not UB but often buggy)
    -fsanitize=null                      # null pointer dereference
    -fsanitize=alignment                 # misaligned pointer
    -fsanitize=bounds                    # array bounds
    -fsanitize=shift                     # invalid shift
    -fsanitize=vla-bound                 # negative VLA size

Example output:

.. code-block:: text

    main.c:10:15: runtime error: signed integer overflow: 
    2147483647 + 1 cannot be represented in type 'int'

UBSan can be combined with ASan:

.. code-block:: bash

    $ gcc -fsanitize=address,undefined -g -o myprogram main.c

MemorySanitizer (MSan)
----------------------

MemorySanitizer detects reads of uninitialized memory, which cause unpredictable
behavior because the value depends on whatever happened to be in memory. These
bugs are difficult to find because the program may appear to work correctly
when uninitialized memory happens to contain benign values. MSan tracks the
initialization state of every bit of memory and reports when uninitialized
values affect program behavior. Only available in Clang.

.. code-block:: bash

    $ clang -fsanitize=memory -g -o myprogram main.c
    $ ./myprogram

    # Track origins of uninitialized values (slower but more helpful)
    $ clang -fsanitize=memory -fsanitize-memory-track-origins -g -o myprogram main.c

Example output:

.. code-block:: text

    ==12345==WARNING: MemorySanitizer: use-of-uninitialized-value
        #0 0x400567 in main main.c:8

    Uninitialized value was created by a heap allocation
        #0 0x7f1234 in malloc
        #1 0x400537 in main main.c:5

Combining Sanitizers
--------------------

Some sanitizers can be combined to catch multiple bug classes in a single build,
while others are mutually exclusive due to conflicting instrumentation:

.. code-block:: bash

    # ASan + UBSan (common combination)
    $ gcc -fsanitize=address,undefined -g -o myprogram main.c

    # ASan + LSan (LSan included by default with ASan on Linux)
    $ gcc -fsanitize=address -g -o myprogram main.c

Cannot combine:

- ASan with TSan (both instrument memory accesses differently)
- ASan with MSan (conflicting shadow memory schemes)
- TSan with MSan

CMake Integration
-----------------

Add sanitizers to CMake builds for easy toggling during development:

.. code-block:: cmake

    # Option to enable sanitizers
    option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
    option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
    option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

    if(ENABLE_ASAN)
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      add_link_options(-fsanitize=address)
    endif()

    if(ENABLE_TSAN)
      add_compile_options(-fsanitize=thread)
      add_link_options(-fsanitize=thread)
    endif()

    if(ENABLE_UBSAN)
      add_compile_options(-fsanitize=undefined)
      add_link_options(-fsanitize=undefined)
    endif()

Build with:

.. code-block:: bash

    $ cmake -DENABLE_ASAN=ON ..
    $ make
