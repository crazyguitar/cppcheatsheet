=========
Pipelines
=========

.. meta::
   :description: CUDA Pipelines for overlapping compute with asynchronous data copies using multi-buffer producer-consumer patterns.
   :keywords: CUDA, pipeline, memcpy_async, producer_acquire, producer_commit, consumer_wait, consumer_release

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Pipelines coordinate multi-buffer producer-consumer patterns to overlap compute
with asynchronous data copies. While one buffer is being processed, another is
being filled - hiding memory latency.

.. code-block:: text

    Without Pipeline:
    [Copy A][Compute A][Copy B][Compute B][Copy C][Compute C]

    With Pipeline (2 stages):
    [Copy A][Copy B   ][Copy C   ]
           [Compute A][Compute B][Compute C]
    ↑ Overlapped execution

Pipelines vs Async Data Copies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipelines and async data copies are related but different abstraction levels:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────┐
    │                    cuda::pipeline                        │
    │  (Multi-stage orchestration: acquire/commit/wait/release)│
    ├──────────────────────────────────────────────────────────┤
    │              Async Data Copy APIs                        │
    │  cuda::memcpy_async, cooperative_groups::memcpy_async    │
    ├──────────────────────────────────────────────────────────┤
    │           Hardware Instructions                          │
    │  LDGSTS (CC 8.0+), TMA (CC 9.0+), STAS (CC 9.0+)         │
    └──────────────────────────────────────────────────────────┘

- **Async Data Copies**: The underlying copy mechanism (global → shared memory).
  Uses LDGSTS (Load Global, Store Shared) instruction - a single PTX op that
  bypasses registers to copy directly. Completion signaled via ``cuda::barrier``
  or pipeline.
- **Pipelines**: Orchestration layer for multi-stage prefetching. Adds state
  machine (acquire/commit/wait/release) to enable double/multi-buffering.

When to use what:

- **Single async copy**: Use ``cuda::memcpy_async`` + ``cuda::barrier``
- **Overlapping load/compute**: Use ``cuda::pipeline`` with ``memcpy_async``
- **CC 9.0+ bulk transfers**: Consider TMA (Tensor Memory Accelerator)

Async Copy with Barrier (No Pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this simpler pattern when you need a single async copy without multi-stage
orchestration. Common scenarios: loading a tile once before a reduction, or
prefetching read-only data at kernel start. The barrier tracks copy completion
without the overhead of pipeline state management.

.. code-block:: cuda

    #include <cuda/barrier>
    #include <cooperative_groups.h>

    __global__ void kernel(float* global_data) {
        namespace cg = cooperative_groups;
        __shared__ float smem[256];
        __shared__ cuda::barrier<cuda::thread_scope_block> bar;

        auto block = cg::this_thread_block();
        if (block.thread_rank() == 0) {
            init(&bar, block.size());  // Initialize barrier
        }
        block.sync();

        // Async copy: global → shared
        cuda::memcpy_async(block, smem, global_data, sizeof(smem), bar);

        bar.arrive_and_wait();  // Wait for copy completion

        // Use smem here...
    }

Quick Reference
---------------

Essential APIs and patterns for CUDA pipelines. Use ``cuda::pipeline`` for
multi-stage prefetching, or ``cuda::barrier`` with ``memcpy_async`` for simpler
single-copy scenarios.

Headers
~~~~~~~

.. code-block:: cuda

    #include <cuda/pipeline>      // cuda::pipeline, memcpy_async
    #include <cuda/barrier>       // cuda::barrier (no pipeline)
    #include <cuda_pipeline.h>    // __pipeline_* primitives

Pipeline Lifecycle
~~~~~~~~~~~~~~~~~~

Thread-scope pipeline is the simplest form - each thread manages its own pipeline
state independently. Use for per-thread prefetching patterns.

.. code-block:: cuda

    // Flow: acquire → copy → commit → wait → use → release
    //
    //   ┌─────────┐   ┌─────────┐   ┌─────────┐
    //   │ acquire │──▶│  copy   │──▶│ commit  │ ← Producer
    //   └─────────┘   └─────────┘   └─────────┘
    //                                    │
    //   ┌─────────┐   ┌─────────┐   ┌────▼────┐
    //   │ release │◀──│  use    │◀──│  wait   │ ← Consumer
    //   └─────────┘   └─────────┘   └─────────┘

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    pipe.producer_acquire();                   // Acquire stage
    cuda::memcpy_async(dst, src, size, pipe);  // Submit async copy
    pipe.producer_commit();                    // Commit stage

    pipe.consumer_wait();                      // Wait for oldest stage
    // ... use data ...
    pipe.consumer_release();                   // Release stage for reuse

Block-Scope Pipeline
~~~~~~~~~~~~~~~~~~~~

Block-scope pipelines synchronize across all threads in a block. Required when
threads cooperatively load shared memory tiles.

.. code-block:: cuda

    // All threads share pipeline state in shared memory
    //
    //   Thread 0 ─┐              ┌─ Thread 0
    //   Thread 1 ─┼─▶ [shared] ◀─┼─ Thread 1
    //   Thread N ─┘   pipeline   └─ Thread N

    constexpr int stages = 2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> shared_state;
    auto pipe = cuda::make_pipeline(block, &shared_state);

Partitioned Pipeline
~~~~~~~~~~~~~~~~~~~~

Separate producer and consumer threads for asymmetric workloads. Producers handle
data loading while consumers process - useful when compute is heavier than copy.

.. code-block:: cuda

    // Threads 0-63: producers    Threads 64-127: consumers
    //   ┌──────────┐               ┌──────────┐
    //   │  load    │──▶ [smem] ──▶│ compute  │
    //   └──────────┘               └──────────┘

    size_t num_producers = block.size() / 2;
    auto pipe = cuda::make_pipeline(block, &shared_state, num_producers);

    // Or explicit roles
    auto role = (threadIdx.x < 64) ? cuda::pipeline_role::producer
                                   : cuda::pipeline_role::consumer;
    auto pipe = cuda::make_pipeline(block, &shared_state, role);

Wait Variants
~~~~~~~~~~~~~

``wait_prior<N>`` keeps N stages in-flight while waiting. Use N > 0 to maintain
overlap when you need to wait but want to keep prefetching ahead.

.. code-block:: cuda

    // wait_prior<0>: wait for ALL stages (same as consumer_wait)
    // wait_prior<1>: keep 1 in-flight, wait for rest
    // wait_prior<2>: keep 2 in-flight, wait for rest
    //
    //   Stages: [0][1][2][3]    wait_prior<2>
    //            ▲  ▲  └──┴── kept in-flight
    //            └──┴── waited

    pipe.consumer_wait();                         // Wait for oldest (tail) stage
    cuda::pipeline_consumer_wait_prior<N>(pipe);  // Wait for all except last N stages

Async Copy Without Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simpler alternative when you only need one async copy without multi-stage
orchestration. Use for single-tile loads or read-only data prefetch.

.. code-block:: cuda

    // Single async copy with barrier (no pipeline overhead)
    //
    //   [init barrier] → [async copy] → [arrive_and_wait] → [use data]

    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    init(&bar, block.size());
    cuda::memcpy_async(block, dst, src, bytes, bar);
    bar.arrive_and_wait();

Primitives API (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~

Lower-level C-style API. Prefer ``cuda::pipeline`` for new code, but primitives
work on older CUDA versions and have less overhead.

.. code-block:: cuda

    __pipeline_memcpy_async(dst, src, size);  // Submit async copy
    __pipeline_commit();                       // Commit current batch
    __pipeline_wait_prior(N);                  // Wait for all except last N (0 = wait all)

Common Patterns
~~~~~~~~~~~~~~~

- Double buffer: ``buf[2][TILE]``, ``stage = t % 2``
- Multi-stage: ``buf[STAGES][TILE]``, ``stage = t % STAGES``
- Early exit: ``pipe.quit(); return;``

Gotchas
~~~~~~~

- ``__syncwarp()`` before commit to avoid warp entanglement
- ``pipe.quit()`` before early return (block-scope)
- Stages = memory_latency / compute_time (typically 2-4)
- LDGSTS (Load Global, Store Shared) requires CC 8.0+, TMA requires CC 9.0+

Double Buffering Pattern
------------------------

:Source: `src/cuda/pipeline <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/pipeline>`_

The most common use case: process one buffer while loading the next. This hides
memory latency by overlapping data transfer with computation. The key insight is
that global memory loads take hundreds of cycles, but with pipelining, those
cycles are spent doing useful compute work instead of waiting.

.. code-block:: cuda

    // Without pipeline (sequential):
    // [Load T0][Proc T0][Load T1][Proc T1][Load T2][Proc T2]
    // ↑ GPU idle during loads
    //
    // With double buffering:
    // Stage 0: [Load T0]         [Load T2]
    // Stage 1:          [Load T1]         [Load T3]
    // Compute:          [Proc T0][Proc T1][Proc T2][Proc T3]
    // ↑ Loads overlap with compute

    __global__ void double_buffer(const float* in, float* out, int n) {
      constexpr int TILE = 256;
      __shared__ float buf[2][TILE];  // Double buffer

      cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

      int tid = threadIdx.x;
      int num_tiles = (n + TILE - 1) / TILE;

      // Prefetch first tile
      pipe.producer_acquire();
      if (tid < TILE && tid < n)
        cuda::memcpy_async(&buf[0][tid], &in[tid], sizeof(float), pipe);
      pipe.producer_commit();

      for (int t = 0; t < num_tiles; t++) {
        int curr = t % 2;
        int next = (t + 1) % 2;

        // Prefetch next tile (if exists)
        if (t + 1 < num_tiles) {
          pipe.producer_acquire();
          int idx = (t + 1) * TILE + tid;
          if (tid < TILE && idx < n)
            cuda::memcpy_async(&buf[next][tid], &in[idx], sizeof(float), pipe);
          pipe.producer_commit();
        }

        // Wait for current tile
        pipe.consumer_wait();
        __syncthreads();

        // Process current tile
        int idx = t * TILE + tid;
        if (tid < TILE && idx < n)
          out[idx] = buf[curr][tid] * 2.0f;

        pipe.consumer_release();
        __syncthreads();
      }
    }

Multi-Stage Pipeline
--------------------

More stages = more latency hiding, but more shared memory. Use 2-4 stages
typically. More stages help when memory latency is high relative to compute time.
The number of stages should match the ratio of memory latency to compute time.

.. code-block:: cuda

    // 4-stage pipeline processing 8 tiles:
    // Stage 0: [T0]       [T4]
    // Stage 1:    [T1]       [T5]
    // Stage 2:       [T2]       [T6]
    // Stage 3:          [T3]       [T7]
    // Compute:          [T0][T1][T2][T3][T4][T5][T6][T7]

    template<int STAGES>
    __global__ void multi_stage(const float* in, float* out, int n) {
      constexpr int TILE = 128;
      __shared__ float buf[STAGES][TILE];

      cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

      int tid = threadIdx.x;
      int num_tiles = (n + TILE - 1) / TILE;

      // Fill pipeline with initial stages
      for (int s = 0; s < STAGES && s < num_tiles; s++) {
        pipe.producer_acquire();
        int idx = s * TILE + tid;
        if (tid < TILE && idx < n)
          cuda::memcpy_async(&buf[s][tid], &in[idx], sizeof(float), pipe);
        pipe.producer_commit();
      }

      // Process tiles
      for (int t = 0; t < num_tiles; t++) {
        int stage = t % STAGES;

        // Prefetch future tile
        if (t + STAGES < num_tiles) {
          pipe.producer_acquire();
          int idx = (t + STAGES) * TILE + tid;
          if (tid < TILE && idx < n)
            cuda::memcpy_async(&buf[stage][tid], &in[idx], sizeof(float), pipe);
          pipe.producer_commit();
        }

        // Wait and process
        pipe.consumer_wait();
        __syncthreads();

        int idx = t * TILE + tid;
        if (tid < TILE && idx < n)
          out[idx] = buf[stage][tid] * 2.0f;

        pipe.consumer_release();
        __syncthreads();
      }
    }

Warp Entanglement
-----------------

Pipeline commits are coalesced within a warp. When threads diverge before commit,
each diverged thread creates a separate stage, spreading operations across many
stages. This causes ``consumer_wait()`` to wait longer than expected because it
waits for the actual (warp-shared) sequence, not the thread's perceived sequence.

.. code-block:: text

    Converged warp (good):
    All 32 threads commit together → 1 stage created

    Diverged warp (bad):
    Thread 0 commits alone → stage 0
    Thread 1 commits alone → stage 1
    ...
    Thread 31 commits alone → stage 31
    consumer_wait() waits for ALL 32 stages!

.. code-block:: cuda

    // BAD: Divergent commit spreads operations across many stages
    if (some_condition) {
      pipe.producer_acquire();
      cuda::memcpy_async(...);
      pipe.producer_commit();  // Each diverged thread commits separately
    }

    // GOOD: Reconverge before commit
    if (some_condition) {
      pipe.producer_acquire();
      cuda::memcpy_async(...);
    }
    __syncwarp();  // Reconverge
    pipe.producer_commit();  // Single coalesced commit

Early Exit
----------

When using block-scope pipelines, threads exiting early must call ``quit()`` to
properly drop out of participation. Otherwise, remaining threads may deadlock
waiting for the exited thread at synchronization points.

.. code-block:: cuda

    if (should_exit) {
      pipe.quit();
      return;
    }

API Comparison
--------------

+---------------------------+--------------------------------+
| cuda::pipeline            | Primitives                     |
+===========================+================================+
| ``producer_acquire()``    | (implicit)                     |
+---------------------------+--------------------------------+
| ``memcpy_async(...,pipe)``| ``__pipeline_memcpy_async()``  |
+---------------------------+--------------------------------+
| ``producer_commit()``     | ``__pipeline_commit()``        |
+---------------------------+--------------------------------+
| ``consumer_wait()``       | ``__pipeline_wait_prior(0)``   |
+---------------------------+--------------------------------+
| ``consumer_release()``    | (implicit)                     |
+---------------------------+--------------------------------+
| wait_prior<N>             | ``__pipeline_wait_prior(N)``   |
+---------------------------+--------------------------------+

Real-World Example: Multi-LoRA Serving
--------------------------------------

vLLM's multi-LoRA implementation uses pipelines to overlap loading LoRA weight
tiles with matrix-vector computation. Each request may use a different LoRA
adapter, so weights are gathered per-batch and pipelined to hide latency.

:Source: `Punica BGMV kernel <https://github.com/punica-ai/punica/blob/master/src/punica/ops/csrc/bgmv/bgmv_impl.cuh>`_
         (`vLLM integration PR #1804 <https://github.com/vllm-project/vllm/pull/1804>`_)

.. code-block:: text

    Multi-tenant LoRA serving:
    ┌─────────────────────────────────────────────────────────┐
    │ Request 1 → LoRA adapter A                              │
    │ Request 2 → LoRA adapter B    →  Batched inference      │
    │ Request 3 → LoRA adapter A       with weight gathering  │
    └─────────────────────────────────────────────────────────┘

    Pipeline hides weight loading latency:
    Stage 0: [Load W tile 0][Load W tile 2]...
    Stage 1:        [Load W tile 1][Load W tile 3]...
    Compute:        [GEMV tile 0  ][GEMV tile 1  ]...

Simplified pattern from the BGMV kernel:

.. code-block:: cuda

    // Double-buffered weight loading for batched LoRA GEMV
    template<int feat_in, int feat_out>
    __global__ void lora_gemv_pipelined(
        float* Y, const float* X, const float* W,
        const int64_t* lora_indices, int batch_size) {

      constexpr int STAGES = 2;
      constexpr int TILE = 128;  // tx * ty * vec_size
      __shared__ float W_shared[STAGES][TILE];
      __shared__ float X_shared[STAGES][TILE];

      auto block = cg::this_thread_block();
      int batch_idx = blockIdx.y;
      int64_t lora_idx = lora_indices[batch_idx];

      auto pipe = cuda::make_pipeline();

      // Kick off first tile load
      pipe.producer_acquire();
      cuda::memcpy_async(W_shared[0] + threadIdx.x,
                         W + lora_idx * feat_out * feat_in + threadIdx.x,
                         sizeof(float), pipe);
      cuda::memcpy_async(X_shared[0] + threadIdx.x,
                         X + batch_idx * feat_in + threadIdx.x,
                         sizeof(float), pipe);
      pipe.producer_commit();

      float y = 0.f;
      for (int tile = 1; tile < feat_in / TILE; tile++) {
        int copy_stage = tile % STAGES;
        int compute_stage = (tile - 1) % STAGES;

        // Async load next tile
        pipe.producer_acquire();
        cuda::memcpy_async(W_shared[copy_stage] + threadIdx.x,
                           W + lora_idx * feat_out * feat_in + tile * TILE + threadIdx.x,
                           sizeof(float), pipe);
        cuda::memcpy_async(X_shared[copy_stage] + threadIdx.x,
                           X + batch_idx * feat_in + tile * TILE + threadIdx.x,
                           sizeof(float), pipe);
        pipe.producer_commit();

        // Compute on previous tile
        pipe.consumer_wait();
        block.sync();
        y += W_shared[compute_stage][threadIdx.x] * X_shared[compute_stage][threadIdx.x];
        // ... warp reduction ...
        pipe.consumer_release();
        block.sync();
      }

      // Final tile
      pipe.consumer_wait();
      // ... final compute and write Y ...
    }

Key techniques from vLLM's implementation:

- **Batched gather**: Each request indexes into different LoRA weights via ``lora_indices``
- **Double buffering**: 2 pipeline stages overlap load/compute
- **Vectorized loads**: Uses ``vec_t`` for coalesced 8-element transfers
- **Warp reduction**: Partial sums reduced within warps using ``__shfl_down_sync``
