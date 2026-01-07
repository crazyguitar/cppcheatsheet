==================
Cooperative Groups
==================

.. meta::
   :description: CUDA Cooperative Groups covering thread synchronization, group partitioning, and collective operations for flexible parallel programming.
   :keywords: CUDA, cooperative groups, __syncthreads, __syncwarp, thread_block, tiled_partition, coalesced_threads, reduce

.. contents:: Table of Contents
    :backlinks: none

Quick Reference
---------------

:Source: `src/cuda/coop-groups-cheatsheet <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-cheatsheet>`_

.. code-block:: cuda

    #include <cooperative_groups.h>
    #include <cooperative_groups/reduce.h>
    namespace cg = cooperative_groups;

    __global__ void kernel() {
      // Create groups
      cg::thread_block block = cg::this_thread_block();           // All threads in block
      cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);  // 32-thread warp
      cg::thread_block_tile<4> tile = cg::tiled_partition<4>(warp);     // 4-thread tile
      cg::coalesced_group active = cg::coalesced_threads();       // Currently active threads

      // Group properties
      block.size();          // Number of threads in group
      block.thread_rank();   // Thread index within group (0 to size-1)
      warp.meta_group_rank();    // Which warp in block (0, 1, 2, ...)
      warp.meta_group_size();    // Number of warps in block

      // Synchronization
      block.sync();          // __syncthreads() equivalent
      warp.sync();           // __syncwarp() equivalent
      tile.sync();           // Sync 4 threads

      // Collective reductions (all threads get result)
      float sum = cg::reduce(warp, val, cg::plus<float>());    // Sum
      float max = cg::reduce(warp, val, cg::greater<float>()); // Max
      float min = cg::reduce(warp, val, cg::less<float>());    // Min
      int all_and = cg::reduce(warp, bits, cg::bit_and<int>()); // Bitwise AND
      int all_or = cg::reduce(warp, bits, cg::bit_or<int>());   // Bitwise OR
      int all_xor = cg::reduce(warp, bits, cg::bit_xor<int>()); // Bitwise XOR

      // Inclusive/Exclusive scan (prefix sum)
      float incl = cg::inclusive_scan(warp, val, cg::plus<float>()); // [a, a+b, a+b+c, ...]
      float excl = cg::exclusive_scan(warp, val, cg::plus<float>()); // [0, a, a+b, ...]

      // Shuffle operations (data exchange within group)
      float broadcasted = warp.shfl(val, 0);           // Broadcast from thread 0
      float from_src = warp.shfl(val, src_lane);       // Get value from src_lane
      float shifted = warp.shfl_down(val, 1);          // Shift values down by 1
      float shifted_up = warp.shfl_up(val, 1);         // Shift values up by 1
      float swapped = warp.shfl_xor(val, 1);           // XOR swap with neighbor

      // Predicates (vote operations)
      bool any_true = warp.any(predicate);       // True if any thread has predicate
      bool all_true = warp.all(predicate);       // True if all threads have predicate
      unsigned ballot = warp.ballot(predicate);  // Bitmask of predicate results

      // Labeled partition (group threads by label value)
      cg::coalesced_group same_label = cg::labeled_partition(warp, label);

      // Thread indexing helpers
      int global_idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
      dim3 block_idx = block.group_index();   // blockIdx equivalent
      dim3 block_dim = block.group_dim();     // blockDim equivalent
      dim3 thread_idx = block.thread_index(); // threadIdx equivalent

      // Elect single thread (useful for single-thread work)
      if (warp.thread_rank() == 0) { /* only one thread executes */ }

      // Match threads with same value
      unsigned match_mask = __match_any_sync(0xffffffff, value);  // Threads with same value
    }

Introduction
------------

Cooperative Groups is a CUDA programming model extension that provides flexible
thread synchronization and collective operations. Before Cooperative Groups,
synchronization was limited to ``__syncthreads()`` (all threads in a block) or
implicit warp synchronization. Cooperative Groups enables explicit synchronization
at any granularity: warps, tiles, thread blocks, or even across multiple blocks.

The key insight is that threads form hierarchical groups, and you can synchronize
or perform collective operations on any group. This makes code more composable
and easier to reason about, especially for complex algorithms that need different
synchronization scopes.

Legacy Synchronization
----------------------

Before Cooperative Groups, CUDA provided these synchronization primitives:

- ``__syncthreads()``: Barrier for all threads in a block. All threads must reach
  this point before any can proceed. Required when threads share data via shared
  memory.

- ``__syncwarp(mask)``: Barrier for threads in a warp specified by the mask.
  Introduced in CUDA 9.0 to make warp-level synchronization explicit after the
  Volta architecture changed warp execution semantics.

.. code-block:: cuda

    // Legacy synchronization
    __shared__ float sdata[256];

    sdata[threadIdx.x] = input[idx];
    __syncthreads();  // All threads must complete store before any reads

    // Warp-level sync (threads 0-31)
    if (threadIdx.x < 32) {
      __syncwarp(0xFFFFFFFF);  // Full warp mask
    }

The problem with legacy primitives is they're not composable. You can't easily
write a function that synchronizes "its" threads without knowing the calling
context. Cooperative Groups solves this by making groups first-class objects.

Thread Groups Hierarchy
-----------------------

:Source: `src/cuda/coop-groups-basic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-basic>`_

Cooperative Groups provides a hierarchy of thread groups:

- ``thread_block``: All threads in the current block (like ``__syncthreads()``)
- ``tiled_partition<N>``: Static partition of a group into tiles of N threads
- ``coalesced_threads()``: Active threads at the current point (dynamic)
- ``this_thread``: Single thread (useful for generic code)

.. code-block:: cuda

    #include <cooperative_groups.h>
    namespace cg = cooperative_groups;

    __global__ void hierarchy_example(float* data, int n) {
      // Get the thread block group
      cg::thread_block block = cg::this_thread_block();

      // Partition into 32-thread tiles (warps)
      cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

      // Partition warp into 4-thread tiles
      cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(warp);

      // Synchronize at different levels
      block.sync();  // All threads in block
      warp.sync();   // All threads in this warp
      tile4.sync();  // All 4 threads in this tile

      // Get thread's rank within each group
      int block_rank = block.thread_rank();  // 0 to blockDim.x-1
      int warp_rank = warp.thread_rank();    // 0 to 31
      int tile_rank = tile4.thread_rank();   // 0 to 3
    }

Group Properties and Methods
----------------------------

Each group object provides useful properties and methods:

.. code-block:: cuda

    cg::thread_block block = cg::this_thread_block();

    block.size();         // Number of threads in group
    block.thread_rank();  // This thread's index within group (0 to size-1)
    block.sync();         // Synchronize all threads in group

    // For tiled partitions
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    warp.size();          // Always 32 for this tile
    warp.thread_rank();   // 0 to 31
    warp.meta_group_size();   // Number of tiles in parent
    warp.meta_group_rank();   // This tile's index in parent

Collective Operations
---------------------

:Source: `src/cuda/coop-groups-reduce <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-reduce>`_

Cooperative Groups provides built-in collective operations that work on any group:

- ``reduce(group, val, op)``: Reduce values across group using binary operator
- ``shfl(group, val, src)``: Broadcast value from source thread
- ``shfl_up/shfl_down``: Shift values within group
- ``any/all``: Predicate operations

.. code-block:: cuda

    #include <cooperative_groups.h>
    #include <cooperative_groups/reduce.h>
    namespace cg = cooperative_groups;

    __global__ void reduce_example(const float* input, float* output, int n) {
      cg::thread_block block = cg::this_thread_block();
      cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      float val = (idx < n) ? input[idx] : 0.0f;

      // Warp-level reduction - no shared memory needed!
      float warp_sum = cg::reduce(warp, val, cg::plus<float>());

      // First thread in each warp has the result
      if (warp.thread_rank() == 0) {
        atomicAdd(output, warp_sum);
      }
    }

Tiled Partition for Warp-Level Programming
------------------------------------------

:Source: `src/cuda/coop-groups-tile <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-tile>`_

``tiled_partition`` creates fixed-size thread groups, commonly used for warp-level
programming. The tile size must be a power of 2 and at most 32.

.. code-block:: cuda

    // Helper function for manual warp/tile reduction
    template <int TileSize, typename T>
    __device__ T tile_reduce_sum(cg::thread_block_tile<TileSize>& tile, T val) {
      for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
      }
      return val;  // Only thread 0 has correct result
    }

    __global__ void warp_reduce(const float* input, float* output, int n) {
      cg::thread_block block = cg::this_thread_block();
      cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      float val = (idx < n) ? input[idx] : 0.0f;

      float sum = tile_reduce_sum(warp, val);

      if (warp.thread_rank() == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank()] = sum;
      }
    }

Coalesced Threads for Divergent Code
------------------------------------

:Source: `src/cuda/coop-groups-coalesced <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-coalesced>`_

When threads in a warp take different branches (e.g., ``if/else``), only some
threads execute each branch. ``coalesced_threads()`` creates a group containing
only the threads that reached that specific point in the code. This lets you
perform collective operations (like reduce) on just those active threads, rather
than the entire warp.

.. code-block:: cuda

    // Without coalesced_threads: must handle inactive threads manually
    if (val > 0) {
      atomicAdd(output, val);  // Each thread does atomic - slow!
    }

    // With coalesced_threads: reduce among active threads first
    if (val > 0) {
      cg::coalesced_group active = cg::coalesced_threads();  // Only threads where val > 0
      int sum = cg::reduce(active, val, cg::plus<int>());    // Sum among active only
      if (active.thread_rank() == 0) {
        atomicAdd(output, sum);  // Single atomic - fast!
      }
    }

.. code-block:: cuda

    __global__ void coalesced_example(const int* input, int* output, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= n) return;

      int val = input[idx];

      // Only process positive values
      if (val > 0) {
        // Get group of threads that took this branch
        cg::coalesced_group active = cg::coalesced_threads();

        // Reduce only among active threads
        int sum = cg::reduce(active, val, cg::plus<int>());

        // First active thread writes result
        if (active.thread_rank() == 0) {
          atomicAdd(output, sum);
        }
      }
    }

Comparison: Legacy vs Cooperative Groups
----------------------------------------

.. code-block:: cuda

    // Legacy reduction (last warp)
    __shared__ float sdata[256];
    // ... load data ...
    __syncthreads();

    // Unrolled warp reduction - implicit warp sync (unsafe on Volta+)
    if (tid < 32) {
      volatile float* smem = sdata;
      smem[tid] += smem[tid + 32];
      smem[tid] += smem[tid + 16];
      smem[tid] += smem[tid + 8];
      smem[tid] += smem[tid + 4];
      smem[tid] += smem[tid + 2];
      smem[tid] += smem[tid + 1];
    }

    // Cooperative Groups reduction - explicit, safe, readable
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    float val = sdata[tid];
    block.sync();

    val = cg::reduce(warp, val, cg::plus<float>());

Benefits of Cooperative Groups:

- **Explicit synchronization**: No relying on implicit warp behavior
- **Composable**: Functions can sync their own groups without global knowledge
- **Portable**: Works correctly across GPU architectures
- **Readable**: Intent is clear from the code

Low-Level PTX Alternatives
--------------------------

:Source: `src/cuda/coop-groups-ptx <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/coop-groups-ptx>`_

For maximum performance, some libraries (like DeepSeek's DeepEP) use inline PTX
assembly instead of cooperative groups. PTX (Parallel Thread Execution) is
NVIDIA's low-level virtual instruction set that sits between CUDA C++ and the
actual GPU machine code (SASS). Using PTX directly provides finer control over
memory ordering, cache behavior, and instruction selection, but sacrifices
portability and readability.

The main reasons to use PTX over cooperative groups:

- **Memory ordering control**: PTX exposes acquire/release semantics for lock-free
  algorithms, allowing precise control over when writes become visible to other
  threads or devices.
- **Cache hints**: PTX can specify L1/L2 cache policies (no-allocate, evict-first)
  to optimize memory access patterns for specific workloads.
- **Compile-time optimization**: Template-based PTX wrappers enable the compiler
  to fully unroll reduction loops and eliminate branches at compile time.
- **Cross-GPU/CPU synchronization**: System-wide fences are needed for multi-GPU
  communication or GPU-CPU synchronization in distributed training.

Memory Fences
~~~~~~~~~~~~~

Memory fences ensure that memory operations before the fence are visible to other
threads after the fence. CUDA provides built-in fence functions, while PTX offers
more granular control with acquire/release semantics.

**Built-in CUDA fence functions:**

.. code-block:: cuda

    __threadfence_block();   // Visible to threads in same block
    __threadfence();         // Visible to all threads on same GPU
    __threadfence_system();  // Visible to all threads + CPU (multi-GPU, pinned memory)

**PTX fence instructions:**

.. code-block:: cuda

    __device__ void memory_fence_cta() {
      asm volatile("fence.acq_rel.cta;" ::: "memory");
    }
    __device__ void memory_fence_gpu() {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __device__ void memory_fence_sys() {
      asm volatile("fence.acq_rel.sys;" ::: "memory");
    }

**Comparison:**

+--------------------------+------------------------+----------------------------------+
| CUDA Built-in            | PTX Equivalent         | Use Case                         |
+==========================+========================+==================================+
| ``__threadfence_block``  | ``fence.acq_rel.cta``  | Shared memory sync within block  |
+--------------------------+------------------------+----------------------------------+
| ``__threadfence``        | ``fence.acq_rel.gpu``  | Global memory sync across blocks |
+--------------------------+------------------------+----------------------------------+
| ``__threadfence_system`` | ``fence.acq_rel.sys``  | Multi-GPU or GPU-CPU sync        |
+--------------------------+------------------------+----------------------------------+

The key difference is that PTX fences support explicit acquire/release semantics,
while CUDA built-ins provide sequentially consistent fences. PTX also allows
combining fences with specific load/store instructions (e.g., ``ld.acquire``,
``st.release``) for finer-grained control in lock-free algorithms.

.. code-block:: cuda

    // Sequential consistency (CUDA built-in): HEAVY
    // Ensures ALL prior memory operations complete before ANY subsequent ones
    data[idx] = value;
    __threadfence();        // Full barrier - waits for everything
    flag = 1;

    // Acquire/Release (PTX): LIGHTER
    // Only orders specific operations, not everything
    data[idx] = value;
    st_release_gpu(&flag, 1);  // Ensures 'data' write completes before 'flag' write

    // Producer-Consumer Example:
    //
    // Thread 0 (Producer)          Thread 1 (Consumer)
    // -------------------          -------------------
    // data = 42;                   while (ld_acquire(&flag) == 0);
    // st_release(&flag, 1);        result = data;  // Guaranteed to see 42
    //        |                              ^
    //        +------------------------------+
    //        release on 'flag' ensures      acquire on 'flag' ensures
    //        'data=42' is visible           'data' read sees the write
    //        before 'flag=1'                that happened before 'flag=1'

Warp Reduction with PTX Shuffle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both cooperative groups and PTX use warp shuffle instructions for reduction, but
the PTX approach allows template-based compile-time unrolling. The ``if constexpr``
pattern ensures the compiler generates only the shuffle instructions needed for
the specified lane count, with no runtime branches. This is particularly useful
when reducing over sub-warp groups (e.g., 4 or 8 threads) where you want to avoid
unnecessary shuffle operations.

.. code-block:: cuda

    // Cooperative Groups reduction
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float sum = cg::reduce(warp, val, cg::plus<float>());

    // PTX-style reduction with compile-time lane count
    template<int kNumLanes, typename T, typename Op>
    __device__ T warp_reduce(T value, Op op) {
      if constexpr (kNumLanes >= 32) value = op(value, __shfl_xor_sync(0xffffffff, value, 16));
      if constexpr (kNumLanes >= 16) value = op(value, __shfl_xor_sync(0xffffffff, value, 8));
      if constexpr (kNumLanes >= 8)  value = op(value, __shfl_xor_sync(0xffffffff, value, 4));
      if constexpr (kNumLanes >= 4)  value = op(value, __shfl_xor_sync(0xffffffff, value, 2));
      if constexpr (kNumLanes >= 2)  value = op(value, __shfl_xor_sync(0xffffffff, value, 1));
      return value;
    }

    // Usage with __device__ lambda (requires --extended-lambda)
    inline constexpr auto ReduceSum = [] __device__(auto a, auto b) { return a + b; };
    float sum = warp_reduce<32>(val, ReduceSum);

Acquire/Release Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~

PTX provides explicit memory ordering semantics for implementing lock-free data
structures and synchronization primitives. The acquire/release model ensures that:

- **Release**: All memory writes before a release store are visible to any thread
  that performs an acquire load of the same location.
- **Acquire**: All memory reads after an acquire load see the writes that happened
  before the corresponding release store.

This is essential for implementing correct spinlocks, producer-consumer queues,
and other synchronization patterns without using expensive full memory barriers.

.. code-block:: cuda

    // Store with release semantics (makes prior writes visible)
    __device__ void st_release(int* ptr, int val) {
      asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
    }

    // Load with acquire semantics (sees prior released writes)
    __device__ int ld_acquire(int* ptr) {
      int ret;
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
      return ret;
    }

    // Spinlock using acquire/release
    __device__ void acquire_lock(int* mutex) {
      while (atomicCAS(mutex, 0, 1) != 0);  // Simplified; real impl uses PTX
    }
    __device__ void release_lock(int* mutex) {
      atomicExch(mutex, 0);
    }

When to Use Each Approach
~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+---------------------------+---------------------------+
| Feature              | Cooperative Groups        | PTX Assembly              |
+======================+===========================+===========================+
| Readability          | High                      | Low                       |
+----------------------+---------------------------+---------------------------+
| Portability          | Across GPU architectures  | May need SM-specific code |
+----------------------+---------------------------+---------------------------+
| Performance          | Good                      | Optimal (hand-tuned)      |
+----------------------+---------------------------+---------------------------+
| Memory ordering      | Implicit                  | Explicit control          |
+----------------------+---------------------------+---------------------------+
| Use case             | Most CUDA code            | HPC, ML frameworks        |
+----------------------+---------------------------+---------------------------+

Use Cooperative Groups for most code. Use PTX only when profiling shows it's
necessary, such as in communication-heavy distributed training kernels where
precise memory ordering and cache control can significantly impact performance.
