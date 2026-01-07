// Cooperative Groups vs PTX - Low-level synchronization primitives
//
// Compares cooperative groups with PTX assembly for warp reduction.
// PTX provides finer control but sacrifices readability and portability.
//
// Warp Reduction Comparison:
// ==========================
//
//   Cooperative Groups (cg::reduce):
//   +----+----+----+----+----+----+----+----+
//   | T0 | T1 | T2 | T3 | .. | T30| T31|     Registers
//   +----+----+----+----+----+----+----+----+
//         \    \    \    \   /    /
//          \    \    \    \ /    /
//           cg::reduce(warp, val, cg::plus<float>())
//                      |
//                      v
//                   [sum]  All threads get result
//
//   PTX __shfl_xor_sync (manual):
//   +----+----+----+----+----+----+----+----+
//   | T0 | T1 | T2 | T3 | .. | T30| T31|
//   +----+----+----+----+----+----+----+----+
//     |    |    |    |         |    |
//     v    v    v    v         v    v
//   __shfl_xor_sync(mask, val, 16)  // T0<->T16, T1<->T17, ...
//   __shfl_xor_sync(mask, val, 8)   // T0<->T8,  T1<->T9, ...
//   __shfl_xor_sync(mask, val, 4)   // ...
//   __shfl_xor_sync(mask, val, 2)
//   __shfl_xor_sync(mask, val, 1)
//                      |
//                      v
//                   [sum]  All threads get result
//
//   Memory Fence Scopes (PTX):
//   +--------------------------------------------------+
//   | fence.acq_rel.sys  | System-wide (CPU + all GPUs)|
//   | fence.acq_rel.gpu  | Single GPU (all SMs)        |
//   | fence.acq_rel.cta  | Single block (CTA)          |
//   +--------------------------------------------------+

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace cg = cooperative_groups;

// Reduction operations as __device__ lambdas
inline constexpr auto ReduceSum = [] __device__(auto a, auto b) { return a + b; };
inline constexpr auto ReduceMax = [] __device__(auto a, auto b) { return a > b ? a : b; };
inline constexpr auto ReduceMin = [] __device__(auto a, auto b) { return a < b ? a : b; };
inline constexpr auto ReduceAnd = [] __device__(auto a, auto b) { return a & b; };
inline constexpr auto ReduceOr = [] __device__(auto a, auto b) { return a | b; };

// PTX-style warp reduction with compile-time unrolling
template <int kNumLanes, typename T, typename Op>
__device__ T warp_reduce_ptx(T value, Op op) {
  constexpr uint32_t mask = 0xffffffff;
  if constexpr (kNumLanes >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
  if constexpr (kNumLanes >= 16) value = op(value, __shfl_xor_sync(mask, value, 8));
  if constexpr (kNumLanes >= 8) value = op(value, __shfl_xor_sync(mask, value, 4));
  if constexpr (kNumLanes >= 4) value = op(value, __shfl_xor_sync(mask, value, 2));
  if constexpr (kNumLanes >= 2) value = op(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

// =============================================================================
// Memory Fence Comparison: CUDA Built-in vs PTX
// =============================================================================
//
// CUDA Built-in          PTX Equivalent           Scope
// ------------------     --------------------     -------------------------
// __threadfence_block()  fence.acq_rel.cta        Block (CTA) - shared mem
// __threadfence()        fence.acq_rel.gpu        GPU - global mem across blocks
// __threadfence_system() fence.acq_rel.sys        System - multi-GPU + CPU
//
// PTX advantage: acquire/release semantics for lighter-weight sync
// =============================================================================

// CUDA built-in fences (sequential consistency)
__device__ __forceinline__ void fence_block_builtin() { __threadfence_block(); }
__device__ __forceinline__ void fence_gpu_builtin() { __threadfence(); }
__device__ __forceinline__ void fence_sys_builtin() { __threadfence_system(); }

// PTX fences (acquire/release semantics)
__device__ __forceinline__ void fence_cta_ptx() { asm volatile("fence.acq_rel.cta;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu_ptx() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_sys_ptx() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

// PTX acquire/release load/store (finer-grained than full fence)
__device__ __forceinline__ void st_release_gpu(int* ptr, int val) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

// Kernel comparing both approaches
__global__ void compare_reduce(const float* input, float* cg_output, float* ptx_output, int n) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? input[idx] : 0.0f;

  // Method 1: Cooperative Groups
  float cg_sum = cg::reduce(warp, val, cg::plus<float>());

  // Method 2: PTX-style with lambda
  float ptx_sum = warp_reduce_ptx<32>(val, ReduceSum);

  // Both should give same result
  if (warp.thread_rank() == 0) {
    int warp_idx = blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank();
    cg_output[warp_idx] = cg_sum;
    ptx_output[warp_idx] = ptx_sum;
  }
}

TEST(CUDA, CoopGroupsPTX) {
  constexpr int n = 256;
  constexpr int threads = 128;
  constexpr int blocks = (n + threads - 1) / threads;
  constexpr int total_warps = blocks * (threads / 32);

  float *h_input, *h_cg, *h_ptx;
  float *d_input, *d_cg, *d_ptx;

  cudaMallocHost(&h_input, n * sizeof(float));
  cudaMallocHost(&h_cg, total_warps * sizeof(float));
  cudaMallocHost(&h_ptx, total_warps * sizeof(float));
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_cg, total_warps * sizeof(float));
  cudaMalloc(&d_ptx, total_warps * sizeof(float));

  for (int i = 0; i < n; i++) {
    h_input[i] = 1.0f;
  }

  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  compare_reduce<<<blocks, threads>>>(d_input, d_cg, d_ptx, n);

  cudaMemcpy(h_cg, d_cg, total_warps * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ptx, d_ptx, total_warps * sizeof(float), cudaMemcpyDeviceToHost);

  // Both methods should produce identical results
  for (int i = 0; i < total_warps; i++) {
    EXPECT_FLOAT_EQ(h_cg[i], 32.0f);
    EXPECT_FLOAT_EQ(h_ptx[i], 32.0f);
    EXPECT_FLOAT_EQ(h_cg[i], h_ptx[i]);
  }

  cudaFree(d_input);
  cudaFree(d_cg);
  cudaFree(d_ptx);
  cudaFreeHost(h_input);
  cudaFreeHost(h_cg);
  cudaFreeHost(h_ptx);
}

// Kernel demonstrating fence comparison: producer-consumer pattern
__global__ void fence_test_kernel(int* flag, int* data, int* result) {
  int tid = threadIdx.x;

  if (tid == 0) {
    // Producer: write data, then signal with fence
    *data = 42;

    // Compare: CUDA built-in vs PTX
    // fence_gpu_builtin();  // __threadfence()
    fence_gpu_ptx();  // fence.acq_rel.gpu

    *flag = 1;
  }

  __syncthreads();

  if (tid == 1) {
    // Consumer: wait for flag, then read data
    while (ld_acquire_gpu(flag) == 0);
    *result = *data;
  }
}

// Kernel demonstrating acquire/release store/load
__global__ void acquire_release_kernel(int* shared_flag, int* shared_data, int* output) {
  int tid = threadIdx.x;

  if (tid == 0) {
    // Producer: write data then release-store the flag
    *shared_data = 100;
    st_release_gpu(shared_flag, 1);  // Release: prior writes visible
  }

  __syncthreads();

  if (tid == 1) {
    // Consumer: acquire-load flag then read data
    while (ld_acquire_gpu(shared_flag) == 0);  // Acquire: sees released writes
    *output = *shared_data;
  }
}

TEST(CUDA, MemoryFences) {
  int *d_flag, *d_data, *d_result;
  int h_result = 0;

  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));
  cudaMemset(d_flag, 0, sizeof(int));
  cudaMemset(d_data, 0, sizeof(int));
  cudaMemset(d_result, 0, sizeof(int));

  fence_test_kernel<<<1, 32>>>(d_flag, d_data, d_result);
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_result, 42);

  cudaFree(d_flag);
  cudaFree(d_data);
  cudaFree(d_result);
}

TEST(CUDA, AcquireRelease) {
  int *d_flag, *d_data, *d_output;
  int h_output = 0;

  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_data, sizeof(int));
  cudaMalloc(&d_output, sizeof(int));
  cudaMemset(d_flag, 0, sizeof(int));
  cudaMemset(d_data, 0, sizeof(int));
  cudaMemset(d_output, 0, sizeof(int));

  acquire_release_kernel<<<1, 32>>>(d_flag, d_data, d_output);
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_output, 100);

  cudaFree(d_flag);
  cudaFree(d_data);
  cudaFree(d_output);
}
