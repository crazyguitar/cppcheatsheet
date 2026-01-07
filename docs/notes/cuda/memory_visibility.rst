=================
Memory Visibility
=================

.. meta::
   :description: Memory visibility and ordering in C++ and CUDA. Covers C++ std::atomic with memory_order options (relaxed, acquire, release, seq_cst), CUDA synchronization primitives (__syncthreads, __threadfence, __threadfence_block, __threadfence_system), cuda::std::atomic from libcu++ with thread_scope (block, device, system), PTX inline assembly for fine-grained control (ld.acquire, st.release, fence.acq_rel), volatile vs atomic vs fence comparison, producer-consumer patterns, spinlock implementation, and multi-GPU visibility. Includes practical examples demonstrating when writes become visible across threads, blocks, GPUs, and CPU.
   :keywords: CUDA, memory visibility, memory fence, acquire release, memory ordering, __threadfence, volatile, atomic, synchronization, std::atomic, cuda::atomic, libcu++, thread_scope, memory_order, PTX, producer consumer, spinlock, multi-GPU

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Memory visibility determines when a write by one thread becomes visible to reads
by other threads. In parallel programming, this is one of the most subtle and
error-prone areas because modern CPUs and GPUs aggressively optimize memory
operations. Without proper synchronization, compilers may reorder instructions,
hardware may cache values in registers or L1 cache, and writes may be delayed
or buffered—leading to bugs that only appear under specific timing conditions
or on certain architectures.

The key insight is that **visibility ≠ ordering ≠ atomicity**:

- **Atomicity**: Operation completes as a single unit (no torn reads/writes)
- **Ordering**: Operations appear in a specific sequence relative to others
- **Visibility**: Writes propagate to other threads/cores/GPUs

C++ Memory Model
----------------

C++ provides memory ordering guarantees through ``std::atomic`` and memory order
specifiers. Understanding these concepts is essential for CUDA programming
because CUDA's memory model builds on similar principles but adds GPU-specific
scopes (block, device, system) to handle the hierarchical nature of GPU memory.

:Source: `src/cpp/memory-visibility <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cpp/memory-visibility>`_

.. code-block:: cpp

    #include <atomic>

    std::atomic<int> flag{0};
    int data = 0;

    // Producer thread
    data = 42;
    flag.store(1, std::memory_order_release);  // Prior writes visible

    // Consumer thread
    while (flag.load(std::memory_order_acquire) == 0);  // Sees prior writes
    assert(data == 42);  // Guaranteed

**Memory Order Options:**

+---------------------------+----------------------------------------------------------+
| Order                     | Guarantee                                                |
+===========================+==========================================================+
| ``memory_order_relaxed``  | No ordering, only atomicity                              |
+---------------------------+----------------------------------------------------------+
| ``memory_order_acquire``  | Reads after this see writes before matching release      |
+---------------------------+----------------------------------------------------------+
| ``memory_order_release``  | Writes before this visible to matching acquire           |
+---------------------------+----------------------------------------------------------+
| ``memory_order_acq_rel``  | Both acquire and release                                 |
+---------------------------+----------------------------------------------------------+
| ``memory_order_seq_cst``  | Total ordering (default, strongest, slowest)             |
+---------------------------+----------------------------------------------------------+

CUDA Memory Hierarchy
---------------------

CUDA has a hierarchical memory architecture where each level has different
visibility characteristics. Writes to shared memory are only visible within
the same block, while writes to global memory can be made visible to all GPU
threads or even the CPU, depending on the fence used.

.. code-block:: text

    +---------------------------------------------------------------+
    |                     System (CPU + GPUs)                       |
    |  +----------------------------------------------------------+ |
    |  |                    GPU Global Memory                     | |
    |  |  +-------------+  +-------------+  +-------------+       | |
    |  |  |   Block 0   |  |   Block 1   |  |   Block N   |       | |
    |  |  | +---------+ |  | +---------+ |  | +---------+ |       | |
    |  |  | | Shared  | |  | | Shared  | |  | | Shared  | |       | |
    |  |  | +---------+ |  | +---------+ |  | +---------+ |       | |
    |  |  | Registers   |  | Registers   |  | Registers   |       | |
    |  |  +-------------+  +-------------+  +-------------+       | |
    |  +----------------------------------------------------------+ |
    +---------------------------------------------------------------+

CUDA Synchronization Primitives
-------------------------------

CUDA provides built-in primitives for synchronization. Barriers like
``__syncthreads()`` both wait for threads and ensure memory visibility, while
fences like ``__threadfence()`` only order memory operations without waiting.
Understanding the difference between volatile, atomics, and fences is critical
for correct inter-thread communication.

:Source: `src/cuda/memory-visibility-primitives <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/memory-visibility-primitives>`_

**Barriers vs Fences:**

Barriers synchronize thread execution AND memory, while fences only order memory.
Use ``__syncthreads()`` when threads need to wait for each other (e.g., after
writing to shared memory). Use ``__threadfence()`` when you only need to ensure
memory writes are visible to other threads without waiting.

+---------------------+------------------+------------------+
| Primitive           | Waits for Threads| Orders Memory    |
+=====================+==================+==================+
| __syncthreads()     | Yes (block)      | Yes (block)      |
+---------------------+------------------+------------------+
| __syncwarp(mask)    | Yes (warp)       | Yes (warp)       |
+---------------------+------------------+------------------+
| __threadfence_block | No               | Yes (block)      |
+---------------------+------------------+------------------+
| __threadfence       | No               | Yes (GPU)        |
+---------------------+------------------+------------------+
| __threadfence_system| No               | Yes (system)     |
+---------------------+------------------+------------------+

.. code-block:: cuda

    // Barrier: waits + orders memory
    __shared__ int data[256];
    data[threadIdx.x] = compute();
    __syncthreads();  // All threads reach here, then continue
    use(data[...]);   // Safe to read any element

    // Fence: orders memory only (no wait)
    global_data[idx] = value;
    __threadfence();  // Ensure write visible before flag
    flag = 1;         // Other threads may not be here yet

**Volatile vs Atomic vs Fence:**

A common mistake is using ``volatile`` for synchronization. While volatile
prevents the compiler from caching values, it provides no hardware guarantees—
writes may still be buffered or reordered by the GPU. Atomics guarantee the
operation itself is indivisible, but ordering depends on the specific operation.
Fences provide explicit ordering without atomicity.

+---------------+------------+------------------+------------------+
| Mechanism     | Atomicity  | Compiler Fence   | Hardware Fence   |
+===============+============+==================+==================+
| volatile      | No         | Yes              | No               |
+---------------+------------+------------------+------------------+
| atomicOp()    | Yes        | Yes              | Varies           |
+---------------+------------+------------------+------------------+
| __threadfence | No         | Yes              | Yes              |
+---------------+------------+------------------+------------------+

.. code-block:: cuda

    // WRONG: volatile doesn't guarantee hardware visibility
    volatile int* flag = ...;
    *data_ptr = 42;
    *flag = 1;  // Other SM may see flag=1 but stale data!

    // CORRECT: use fence
    *data_ptr = 42;
    __threadfence();
    *flag = 1;  // Other SM sees data=42 when flag=1

cuda::std::atomic (libcu++)
---------------------------

CUDA's libcu++ library provides ``cuda::atomic``, which mirrors C++ ``std::atomic``
but adds explicit thread scope control. This allows you to specify the visibility
scope (block, device, or system) and use the same memory order semantics as C++.
This is the recommended approach for portable, readable synchronization code.

:Source: `src/cuda/memory-visibility-libcu <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/memory-visibility-libcu>`_

**Thread Scopes:**

.. code-block:: cuda

    #include <cuda/atomic>

    // Block scope - visible to threads in same block (fastest)
    cuda::atomic<int, cuda::thread_scope_block> blk_atomic{0};

    // Device scope - visible to all threads on GPU (default)
    cuda::atomic<int, cuda::thread_scope_device> gpu_atomic{0};

    // System scope - visible to GPU + CPU (slowest)
    cuda::atomic<int, cuda::thread_scope_system> sys_atomic{0};

**Memory Orders (same as C++):**

.. code-block:: cuda

    // Relaxed - atomicity only, no ordering
    val = a.load(cuda::std::memory_order_relaxed);
    a.store(1, cuda::std::memory_order_relaxed);
    a.fetch_add(1, cuda::std::memory_order_relaxed);

    // Acquire - sees writes before matching release
    val = a.load(cuda::std::memory_order_acquire);

    // Release - prior writes visible to acquire
    a.store(1, cuda::std::memory_order_release);

    // Acquire-Release - both (for read-modify-write)
    old = a.exchange(1, cuda::std::memory_order_acq_rel);

    // Sequential consistency - total ordering (default, slowest)
    val = a.load();  // seq_cst by default

**Common Operations:**

+----------------------------------+----------------------------------+
| cuda::atomic                     | CUDA C-style                     |
+==================================+==================================+
| ``a.store(42)``                  | ``a = 42`` (non-atomic)          |
+----------------------------------+----------------------------------+
| ``a.load()``                     | ``a`` (non-atomic)               |
+----------------------------------+----------------------------------+
| ``a.exchange(1)``                | ``atomicExch(&a, 1)``            |
+----------------------------------+----------------------------------+
| ``a.fetch_add(1)``               | ``atomicAdd(&a, 1)``             |
+----------------------------------+----------------------------------+
| ``a.fetch_sub(1)``               | ``atomicSub(&a, 1)``             |
+----------------------------------+----------------------------------+
| ``a.fetch_and(mask)``            | ``atomicAnd(&a, mask)``          |
+----------------------------------+----------------------------------+
| ``a.fetch_or(mask)``             | ``atomicOr(&a, mask)``           |
+----------------------------------+----------------------------------+
| ``a.fetch_xor(mask)``            | ``atomicXor(&a, mask)``          |
+----------------------------------+----------------------------------+
| ``a.fetch_min(v)``               | ``atomicMin(&a, v)``             |
+----------------------------------+----------------------------------+
| ``a.fetch_max(v)``               | ``atomicMax(&a, v)``             |
+----------------------------------+----------------------------------+
| ``a.compare_exchange_strong(e,d)``| ``atomicCAS(&a, e, d)``         |
+----------------------------------+----------------------------------+

Note: CUDA C-style atomics don't support explicit memory ordering—they use
device-scope with unspecified ordering. Use ``cuda::atomic`` when you need
specific scopes or memory orders.

PTX Memory Operations
---------------------

For maximum control over memory ordering, you can use PTX inline assembly.
PTX provides explicit acquire/release semantics with configurable scopes (cta,
gpu, sys). This is useful when you need finer-grained control than what CUDA
built-ins provide, or when implementing lock-free data structures like spinlocks.

:Source: `src/cuda/memory-visibility-ptx <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/memory-visibility-ptx>`_

Reference: `DeepEP utils.cuh <https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh>`_

**Load/Store with Ordering:**

PTX load/store instructions can include ordering semantics directly. An acquire
load ensures subsequent operations see writes that happened before the matching
release store. This is more efficient than using separate fence instructions.

.. code-block:: cuda

    // GPU scope
    __device__ int ld_acquire_gpu(const int* ptr) {
      int ret;
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
      return ret;
    }

    __device__ void st_release_gpu(int* ptr, int val) {
      asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
    }

    // System scope (multi-GPU / CPU)
    __device__ int ld_acquire_sys(const int* ptr) {
      int ret;
      asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
      return ret;
    }

    __device__ void st_release_sys(int* ptr, int val) {
      asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
    }

**Fences:**

PTX fences provide standalone memory barriers without being tied to a specific
load or store. Use ``fence.acq_rel`` to ensure all prior writes are visible
before subsequent reads. The scope suffix controls visibility range.

.. code-block:: cuda

    __device__ void fence_cta() { asm volatile("fence.acq_rel.cta;" ::: "memory"); }
    __device__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
    __device__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

**PTX Fence vs CUDA Built-in Fence:**

CUDA's ``__threadfence()`` provides sequential consistency—a stronger guarantee
that orders all memory operations. PTX ``fence.acq_rel`` provides acquire-release
semantics which is sufficient for most synchronization patterns and may be faster
on some architectures. In practice, the difference is often negligible, but PTX
gives you explicit control when optimizing hot paths.

+----------------------+---------------------------+---------------------------+
| Aspect               | ``__threadfence_*()``     | ``fence.acq_rel.*``       |
+======================+===========================+===========================+
| Ordering             | Sequential consistency    | Acquire-release           |
+----------------------+---------------------------+---------------------------+
| Strength             | Stronger (all ops ordered)| Weaker (only acq/rel)     |
+----------------------+---------------------------+---------------------------+
| Performance          | Potentially slower        | Potentially faster        |
+----------------------+---------------------------+---------------------------+
| Use case             | Simple, safe default      | Fine-tuned performance    |
+----------------------+---------------------------+---------------------------+

**PTX Spinlock:**

A spinlock protects critical sections where only one thread can execute at a
time. Use ``atom.cas`` with acquire semantics to take the lock (ensuring
subsequent reads see prior writes), and ``atom.exch`` with release semantics
to release it (ensuring prior writes are visible to the next lock holder).

.. code-block:: cuda

    __device__ void acquire_lock(int* mutex) {
      int ret;
      do {
        asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(ret) : "l"(mutex), "r"(0), "r"(1) : "memory");
      } while (ret != 0);
    }

    __device__ void release_lock(int* mutex) {
      int ret;
      asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;"
                   : "=r"(ret) : "l"(mutex), "r"(0) : "memory");
    }

Common Patterns
---------------

The producer-consumer pattern is fundamental to GPU programming. One kernel (or
thread) produces data and signals completion via a flag, while another waits for
the flag and then reads the data. Without proper memory ordering, the consumer
may see the flag set but read stale data.

:Source: `src/cuda/memory-visibility-patterns <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/memory-visibility-patterns>`_

**Producer-Consumer (single GPU):**

Three approaches with different trade-offs. Method 1 uses CUDA built-ins and is
the simplest but requires two fences. Method 2 uses libcu++ atomics which is
portable and readable with C++-style syntax. Method 3 uses PTX acquire/release
which is the lightest weight because ordering is built into the load/store.

.. code-block:: cuda

    // Method 1: __threadfence + atomic (simple, safe)
    __global__ void producer() {
      data = 42;
      __threadfence();
      atomicExch(&flag, 1);
    }

    __global__ void consumer(int* result) {
      while (atomicAdd(&flag, 0) == 0);
      __threadfence();
      *result = data;
    }

    // Method 2: cuda::atomic (portable, readable)
    __device__ cuda::atomic<int, cuda::thread_scope_device> flag{0};
    __global__ void producer() {
      data = 42;
      flag.store(1, cuda::std::memory_order_release);
    }

    __global__ void consumer(int* result) {
      while (flag.load(cuda::std::memory_order_acquire) == 0);
      *result = data;
    }

    // Method 3: PTX acquire/release (lightest weight)
    __global__ void producer() {
      data = 42;
      st_release_gpu(&flag, 1);
    }

    __global__ void consumer(int* result) {
      while (ld_acquire_gpu(&flag) == 0);
      *result = data;
    }

**Multi-GPU Communication:**

For communication between GPUs or between GPU and CPU, use system scope. The
data must be in pinned (page-locked) host memory or P2P-enabled device memory.
System scope fences are expensive, so minimize their use in performance-critical
code.

.. code-block:: cuda

    // GPU 0: Write to pinned host memory
    *host_data = result;
    __threadfence_system();  // Or st_release_sys()
    *host_flag = 1;

    // GPU 1: Read from pinned host memory
    while (*host_flag == 0);
    __threadfence_system();  // Or ld_acquire_sys()
    use(*host_data);

Quick Reference
---------------

Use the narrowest scope possible for best performance—block scope is much faster
than system scope. Prefer ``cuda::atomic`` for readability; use PTX only when
optimizing hot paths.

**When to Use What:**

+---------------------------+------------------------------------------------+
| Scenario                  | Recommended Approach                           |
+===========================+================================================+
| Intra-block sync          | __syncthreads() or cuda::atomic<block>         |
+---------------------------+------------------------------------------------+
| Inter-block on same GPU   | cuda::atomic<device> or PTX acquire/release    |
+---------------------------+------------------------------------------------+
| Multi-GPU / CPU-GPU       | cuda::atomic<system> or PTX .sys scope         |
+---------------------------+------------------------------------------------+
| Critical section (block)  | PTX spinlock with acquire/release              |
+---------------------------+------------------------------------------------+
