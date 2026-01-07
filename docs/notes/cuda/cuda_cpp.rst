========
CUDA C++
========

.. meta::
   :description: libcu++ C++ Standard Library for CUDA - atomics, barriers, semaphores, latches, and memory ordering.
   :keywords: CUDA, libcu++, atomic, barrier, semaphore, latch, memory order, thread scope

.. contents:: Table of Contents
    :backlinks: none

libcu++ is NVIDIA's implementation of the C++ Standard Library for CUDA. It
provides thread-safe, GPU-compatible versions of standard library components
including atomics, barriers, semaphores, and memory ordering primitives. Unlike
raw CUDA intrinsics, libcu++ offers portable abstractions that work across
different GPU architectures and can even compile for CPU targets, making code
more maintainable and easier to reason about.

The library uses C++ memory model semantics (``memory_order_acquire``,
``memory_order_release``, etc.) which map to appropriate PTX instructions on
the GPU. This allows writing correct concurrent code without manually managing
memory fences or understanding PTX details. For detailed coverage of memory
visibility, fences, and PTX-level control, see :doc:`cuda_memory_visibility`.

Atomic Operations
-----------------

:Source: `src/cuda/libcuxx-atomic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/libcuxx-atomic>`_

Atomics provide thread-safe read-modify-write operations. libcu++ atomics
support all standard memory orderings and work across thread scopes (block,
device, system). Use ``cuda::atomic`` for device-wide visibility or
``cuda::atomic_ref`` to wrap existing memory locations.

.. code-block:: cuda

    #include <cuda/atomic>

    // Device-scope atomic counter
    __device__ cuda::atomic<int, cuda::thread_scope_device> counter{0};

    __global__ void increment_kernel() {
      counter.fetch_add(1, cuda::memory_order_relaxed);
    }

    // Atomic with acquire-release for synchronization
    __device__ cuda::atomic<int, cuda::thread_scope_device> flag{0};
    __device__ int data;

    __global__ void producer() {
      data = 42;                                          // Write data
      flag.store(1, cuda::memory_order_release);          // Release: data visible before flag
    }

    __global__ void consumer() {
      while (flag.load(cuda::memory_order_acquire) == 0); // Acquire: see data after flag
      int x = data;                                       // Guaranteed to see 42
    }

    // Atomic reference to existing memory
    __global__ void atomic_ref_kernel(int* arr, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
        cuda::atomic_ref<int, cuda::thread_scope_device> ref(arr[idx]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      }
    }

Barriers
--------

:Source: `src/cuda/libcuxx-barrier <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/libcuxx-barrier>`_

Barriers synchronize groups of threads, ensuring all threads reach a point
before any proceed. libcu++ provides ``cuda::barrier`` which supports both
arrival/wait patterns and the more efficient ``arrive_and_wait``. Barriers
can also track completion of async operations like ``memcpy_async``.

.. code-block:: cuda

    #include <cuda/barrier>
    #include <cooperative_groups.h>

    namespace cg = cooperative_groups;

    __global__ void barrier_kernel() {
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;

      auto block = cg::this_thread_block();
      if (block.thread_rank() == 0) {
        init(&bar, block.size());  // Initialize with thread count
      }
      block.sync();

      // Phase 1: all threads do work
      // ...

      bar.arrive_and_wait();  // Synchronize

      // Phase 2: all threads see Phase 1 results
    }

    // Split arrive/wait for overlapping work
    __global__ void split_barrier_kernel() {
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;
      __shared__ int shared_data[256];

      auto block = cg::this_thread_block();
      if (block.thread_rank() == 0) init(&bar, block.size());
      block.sync();

      // Arrive early, do independent work, then wait
      auto token = bar.arrive();       // Signal arrival
      // ... do work that doesn't need barrier ...
      bar.wait(std::move(token));      // Wait for others
    }

Semaphores
----------

:Source: `src/cuda/libcuxx-semaphore <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/libcuxx-semaphore>`_

Semaphores control access to limited resources. ``counting_semaphore`` allows
up to N concurrent accesses, while ``binary_semaphore`` (N=1) acts as a mutex.
Use for producer-consumer patterns or limiting concurrent operations. For more
detail, see :doc:`cuda_memory_visibility`.

.. code-block:: cuda

    #include <cuda/semaphore>

    // Limit concurrent access to 4 threads
    __device__ cuda::counting_semaphore<cuda::thread_scope_device, 4> sem{4};

    __global__ void limited_access_kernel() {
      sem.acquire();  // Block if 4 threads already inside
      // ... critical section with limited concurrency ...
      sem.release();
    }

    // Binary semaphore as mutex
    __device__ cuda::binary_semaphore<cuda::thread_scope_device> mtx{1};

    __global__ void mutex_kernel(int* shared_counter) {
      mtx.acquire();
      (*shared_counter)++;  // Only one thread at a time
      mtx.release();
    }

Latches
-------

:Source: `src/cuda/libcuxx-latch <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/libcuxx-latch>`_

A latch is a single-use barrier. Threads call ``count_down()`` to decrement
the counter, and ``wait()`` blocks until the counter reaches zero. Unlike
barriers, latches cannot be reused after reaching zero. For more detail, see
:doc:`cuda_memory_visibility`.

.. code-block:: cuda

    #include <cuda/latch>

    __global__ void latch_kernel() {
      __shared__ cuda::latch<cuda::thread_scope_block> lat;

      auto block = cg::this_thread_block();
      if (block.thread_rank() == 0) {
        init(&lat, block.size());
      }
      block.sync();

      // Each thread does work then counts down
      // ... work ...
      lat.count_down();

      // Wait for all threads to finish
      lat.wait();
    }

    // Arrive and wait in one call
    lat.arrive_and_wait();  // count_down() + wait()

Memory Order
------------

libcu++ supports C++ memory orderings that control visibility of memory
operations across threads. Understanding these is crucial for writing correct
lock-free code. See :doc:`cuda_memory_visibility` for detailed coverage of CUDA
memory model.

.. code-block:: cuda

    #include <cuda/atomic>

    // memory_order_relaxed: No ordering guarantees, fastest
    // Use for counters where order doesn't matter
    __device__ cuda::atomic<int, cuda::thread_scope_device> counter{0};
    counter.fetch_add(1, cuda::memory_order_relaxed);

    // memory_order_acquire: Reads after this see writes before matching release
    // memory_order_release: Writes before this visible after matching acquire
    // Use for producer-consumer synchronization
    __device__ cuda::atomic<bool, cuda::thread_scope_device> ready{false};
    __device__ int payload;

    // Producer                              // Consumer
    payload = 42;                            while (!ready.load(cuda::memory_order_acquire));
    ready.store(true, memory_order_release); int x = payload;  // Sees 42

    // memory_order_acq_rel: Both acquire and release
    // Use for read-modify-write that both reads and publishes
    counter.fetch_add(1, cuda::memory_order_acq_rel);

    // memory_order_seq_cst: Total ordering, slowest but simplest
    // Use when unsure or for debugging
    counter.fetch_add(1, cuda::memory_order_seq_cst);

Thread Scopes
-------------

Thread scopes define the visibility domain for synchronization primitives.
Narrower scopes can be more efficient but only synchronize within that scope.

.. code-block:: cuda

    #include <cuda/atomic>

    // thread_scope_thread: Single thread (rarely useful)
    cuda::atomic<int, cuda::thread_scope_thread> local_atomic;

    // thread_scope_block: Threads in same block (uses shared memory)
    __shared__ cuda::atomic<int, cuda::thread_scope_block> block_atomic;

    // thread_scope_device: All threads on GPU (uses global memory)
    __device__ cuda::atomic<int, cuda::thread_scope_device> device_atomic;

    // thread_scope_system: GPU + CPU (for unified memory)
    // Slowest but works across host and device
    cuda::atomic<int, cuda::thread_scope_system>* system_atomic;
    cudaMallocManaged(&system_atomic, sizeof(*system_atomic));

Pipeline with Barrier
---------------------

:Source: `src/cuda/libcuxx-barrier <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/libcuxx-barrier>`_

Combine barriers with ``memcpy_async`` for efficient pipelining. The barrier
tracks completion of async copies, allowing overlap of memory transfers with
computation. See :doc:`cuda_pipelines` for more pipeline patterns.

.. code-block:: cuda

    #include <cuda/barrier>
    #include <cuda/pipeline>
    #include <cooperative_groups.h>

    __global__ void pipeline_with_barrier(const float* global_in, float* global_out, int n) {
      __shared__ float smem[256];
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;

      auto block = cg::this_thread_block();
      if (block.thread_rank() == 0) init(&bar, block.size());
      block.sync();

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < n) {
        // Async copy with barrier tracking
        cuda::memcpy_async(&smem[threadIdx.x], &global_in[idx], sizeof(float), bar);
      }

      bar.arrive_and_wait();  // Wait for all copies to complete

      // Process data in shared memory
      if (idx < n) {
        global_out[idx] = smem[threadIdx.x] * 2.0f;
      }
    }

Quick Reference
---------------

.. code-block:: cuda

    // Headers
    #include <cuda/atomic>           // atomic, atomic_ref
    #include <cuda/barrier>          // barrier
    #include <cuda/semaphore>        // counting_semaphore, binary_semaphore
    #include <cuda/latch>            // latch
    #include <cuda/pipeline>         // pipeline, memcpy_async

    // Thread scopes (narrower = faster)
    cuda::thread_scope_thread        // Single thread
    cuda::thread_scope_block         // Same block (shared memory)
    cuda::thread_scope_device        // All GPU threads (global memory)
    cuda::thread_scope_system        // GPU + CPU (unified memory)

    // Memory orders (weaker = faster)
    cuda::memory_order_relaxed       // No ordering
    cuda::memory_order_acquire       // Reads see prior releases
    cuda::memory_order_release       // Writes visible to acquires
    cuda::memory_order_acq_rel       // Both acquire and release
    cuda::memory_order_seq_cst       // Total ordering

    // Atomic operations
    atomic.load(order)               // Read
    atomic.store(val, order)         // Write
    atomic.exchange(val, order)      // Swap, return old
    atomic.fetch_add(val, order)     // Add, return old
    atomic.fetch_sub(val, order)     // Subtract, return old
    atomic.fetch_and(val, order)     // Bitwise AND, return old
    atomic.fetch_or(val, order)      // Bitwise OR, return old
    atomic.fetch_xor(val, order)     // Bitwise XOR, return old
    atomic.compare_exchange_weak(expected, desired, order)
    atomic.compare_exchange_strong(expected, desired, order)

    // Barrier operations
    init(&bar, count)                // Initialize (once per barrier)
    bar.arrive()                     // Signal arrival, return token
    bar.wait(token)                  // Wait for phase completion
    bar.arrive_and_wait()            // arrive() + wait()

    // Semaphore operations
    sem.acquire()                    // Decrement, block if zero
    sem.release()                    // Increment
    sem.try_acquire()                // Non-blocking acquire

    // Latch operations
    init(&lat, count)                // Initialize
    lat.count_down()                 // Decrement counter
    lat.wait()                       // Block until zero
    lat.arrive_and_wait()            // count_down() + wait()
