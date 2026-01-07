======
Memory
======

.. meta::
   :description: Memory alignment in C covering why alignment matters, different alignment sizes, and performance benchmarks comparing aligned vs unaligned access.
   :keywords: memory alignment, cache line, SIMD, aligned access, unaligned access, performance, alignas, alignof, C11

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Memory alignment refers to placing data at memory addresses that are multiples of
the data's size or a specified boundary. Modern CPUs are optimized for aligned
memory access, and misaligned access can cause performance penalties or even
hardware exceptions on some architectures.

Why Memory Alignment Matters
----------------------------

CPUs access memory in fixed-size chunks (typically 4 or 8 bytes). When data
crosses these boundaries, the CPU must perform multiple memory operations:

1. **Hardware efficiency**: CPUs fetch data in cache-line-sized blocks (typically
   64 bytes). Aligned data ensures single-fetch access.

2. **Atomic operations**: Many atomic instructions require aligned operands.
   Misaligned atomics may not be atomic at all.

3. **SIMD instructions**: Vector operations (SSE, AVX) require or strongly prefer
   aligned data (16, 32, or 64-byte boundaries).

4. **Cache utilization**: Aligned structures pack efficiently into cache lines,
   reducing cache misses and false sharing in multithreaded code.

Different Alignment Sizes
-------------------------

:Source: `src/c/memory-alignment <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/memory-alignment>`_

Alignment follows a hierarchical relationship: higher alignment guarantees all
lower alignments, but not vice versa. An address aligned to 64 bytes is
automatically aligned to 32, 16, 8, 4, 2, and 1 bytes because any number
divisible by 64 is also divisible by all its factors. However, an 8-byte
aligned address (like ``0x18`` = 24) may not be 16-byte aligned since
24 is not divisible by 16.

- 64-byte aligned → also 32, 16, 8, 4, 2, 1-byte aligned
- 16-byte aligned → also 8, 4, 2, 1-byte aligned
- 8-byte aligned → NOT necessarily 16-byte aligned

.. code-block:: c

    // Address 0x40 (64): 64%64=0 ✓, 64%16=0 ✓, 64%8=0 ✓
    // Address 0x18 (24): 24%16=8 ✗, 24%8=0 ✓

Different alignment requirements serve different purposes:

- **8-byte alignment**: Natural alignment for 64-bit types such as ``double``
  and pointers on 64-bit systems. This ensures the CPU can fetch the entire
  value in a single memory access without crossing a word boundary.

- **16-byte alignment**: Required for SSE (128-bit) SIMD operations on x86.
  This is also the standard stack alignment on x86-64 as specified by the
  System V ABI, ensuring function calls maintain proper alignment for vector
  operations.

- **32-byte alignment**: Required for AVX (256-bit) SIMD operations. AVX
  instructions operating on misaligned data may incur performance penalties
  or require separate load instructions.

- **64-byte alignment**: Matches the cache line size on most modern x86 and
  ARM processors. Aligning data structures to cache line boundaries prevents
  false sharing in multithreaded code, where different threads modifying
  adjacent data would otherwise invalidate each other's cache lines.

- **Page alignment (4096 bytes typical)**: Required for memory-mapped I/O,
  ``mmap()``, and DMA buffers. The operating system manages memory in pages,
  so operations like mapping files or allocating huge pages require page-aligned
  addresses. Query the system page size with ``sysconf(_SC_PAGESIZE)``.

.. code-block:: c

    #include <stdalign.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <unistd.h>

    int main(void) {
      // Natural alignment
      printf("alignof(char)   = %zu\n", alignof(char));    // 1
      printf("alignof(int)    = %zu\n", alignof(int));     // 4
      printf("alignof(double) = %zu\n", alignof(double));  // 8

      // System page size
      const size_t page_size = sysconf(_SC_PAGESIZE);
      printf("page size       = %zu\n", page_size);        // 4096 (typical)

      // Custom alignment
      alignas(16) int simd_ready = 42;
      alignas(64) char cache_line[64];

      printf("&simd_ready %% 16 = %zu\n",
             (uintptr_t)&simd_ready % 16);  // 0
      printf("&cache_line %% 64 = %zu\n",
             (uintptr_t)&cache_line % 64);  // 0
    }

.. code-block:: console

    $ ./memory-alignment
    alignof(char)   = 1
    alignof(int)    = 4
    alignof(double) = 8
    page size       = 4096
    &simd_ready % 16 = 0
    &cache_line % 64 = 0

Align Up Technique
------------------

A common pattern is aligning a pointer or size up to the next alignment boundary.
The bitwise formula ``(ptr + align - 1) & ~(align - 1)`` rounds up to the nearest
multiple of ``align`` (which must be a power of 2). This works because ``~(align - 1)``
creates a mask that clears the lower bits, and adding ``align - 1`` first ensures
we round up rather than down.

.. code-block:: cpp

    // Align pointer up to boundary (align must be power of 2)
    constexpr void* align_up(void* ptr, size_t align) noexcept {
      return (void*)(((uintptr_t)ptr + align - 1) & ~(align - 1));
    }

    // Align size up to boundary
    constexpr size_t align_up(size_t size, size_t align) noexcept {
      return (size + align - 1) & ~(align - 1);
    }

    // Example usage for buffer allocation
    void* raw = malloc(size + align - 1);      // allocate extra for alignment
    void* aligned = align_up(raw, align);      // get aligned pointer within buffer

    // Page-aligned allocation
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t alloc_size = align_up(size, page_size);

    /*
    Buffer Allocation with Alignment
    =================================

    1. Allocate extra space:

       raw = malloc(size + align - 1)
       |
       v
       +----------------------------------------------------------+
       | extra padding (up to align-1) |       usable size        |
       +----------------------------------------------------------+
       ^
       raw (may be unaligned)

    2. Align pointer up:

       aligned = align_up(raw, align)
       |
       v
       +----------------------------------------------------------+
       | wasted |              aligned buffer                     |
       +----------------------------------------------------------+
                ^
                aligned (guaranteed aligned to 'align' boundary)
    */

Structure Alignment and Padding
-------------------------------

:Source: `src/c/struct-alignment <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/struct-alignment>`_

Compilers automatically insert padding bytes between struct members to ensure each
member is naturally aligned. This is necessary because accessing misaligned data
can be slower or cause hardware faults on some architectures. The struct's total
size is also padded to a multiple of its largest member's alignment, ensuring
arrays of structs maintain proper alignment.

.. code-block:: cpp

    #include <cstddef>
    #include <cstdio>

    struct Bad {
      char a;      // 1 byte
                   // 3 bytes padding (to align int)
      int b;       // 4 bytes
      char c;      // 1 byte
                   // 3 bytes padding (to align struct size to 4)
    };             // Total: 12 bytes

    struct Good {
      int b;       // 4 bytes
      char a;      // 1 byte
      char c;      // 1 byte
                   // 2 bytes padding
    };             // Total: 8 bytes

    struct Packed {
      char a;
      int b;
      char c;
    } __attribute__((packed));  // Total: 6 bytes (no padding, may be slower)

    int main() {
      printf("sizeof(Bad)    = %zu\n", sizeof(Bad));     // 12
      printf("sizeof(Good)   = %zu\n", sizeof(Good));    // 8
      printf("sizeof(Packed) = %zu\n", sizeof(Packed));  // 6

      printf("offsetof(Bad, a) = %zu\n", offsetof(Bad, a));  // 0
      printf("offsetof(Bad, b) = %zu\n", offsetof(Bad, b));  // 4
      printf("offsetof(Bad, c) = %zu\n", offsetof(Bad, c));  // 8
    }

.. code-block:: console

    $ ./struct-alignment
    sizeof(Bad)    = 12
    sizeof(Good)   = 8
    sizeof(Packed) = 6
    offsetof(Bad, a) = 0
    offsetof(Bad, b) = 4
    offsetof(Bad, c) = 8

To minimize padding, order struct members from largest to smallest alignment.
Use ``__attribute__((packed))`` only when space is critical and you accept the
performance penalty of misaligned access.

Benchmark
---------

:Source: `src/c/memory-alignment <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/memory-alignment>`_

This benchmark measures memory access throughput by summing 1M integers at various
byte offsets from a 64-byte aligned base address. By starting the array at offsets
0, 1, 2, 4, 8, 16, and 32 bytes, we can observe how different alignment boundaries
affect performance on modern hardware.

.. code-block:: console

    $ ./memory-alignment-bench
    Memory Alignment Performance Benchmark
    Array size: 1048576 integers
    Iterations: 100

          Offset      Time (ms)     Aligned to
    ------------------------------------------
               0           0.45        64-byte
               1           0.37      unaligned
               2           0.36      unaligned
               4           0.35         4-byte
               8           0.34         8-byte
              16           0.37        16-byte
              32           0.40        32-byte

On modern CPUs (x86-64, Apple Silicon), the benchmark shows negligible difference
between aligned and unaligned access for scalar code. The hardware handles
misalignment transparently. However, alignment remains critical for SIMD operations,
atomics, and portability to stricter architectures.
