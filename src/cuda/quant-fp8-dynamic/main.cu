// FP8 Quantization - Dynamic Per-Tensor Scale
//
// This example demonstrates dynamic FP8 quantization where the scale is
// computed at runtime from the actual tensor values. This is used for
// activation quantization in LLM inference (vLLM, TensorRT-LLM) where
// activation ranges change per input.
//
// Two-pass approach (same as vLLM):
//   Pass 1: parallel reduction to find absmax across all elements
//   Pass 2: quantize using scale = absmax / fp8_max
//
// Key concepts:
// - Shared memory reduction to find per-tensor absmax
// - atomicMax across blocks for global absmax (float via int reinterpret)
// - Scale is written to device memory between passes
// - Two-kernel approach avoids global barrier
//
// Graph structure:
//   [absmax reduction] → [quantize with computed scale]
//
// Reference: vllm/csrc/quantization/w8a8/fp8/common.cu
//            (segmented_max_reduction + scaled_fp8_quant_kernel_strided_dynamic)
//
// Pitfalls:
// - atomicMax on float requires int reinterpretation trick (no native atomicMaxFloat)
// - Scale must be initialized to 0 before the reduction kernel
// - All-zero input produces scale=0; guard with epsilon

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>

constexpr float FP8_E4M3_MAX = 448.0f;

// Float atomicMax via int reinterpretation (works for non-negative values)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  return (value >= 0)
             ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
             : __uint_as_float(
                   atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

// Pass 1: find absmax across the tensor, write scale = absmax / fp8_max
__global__ void find_absmax_and_scale(
    float* __restrict__ scale,
    const float* __restrict__ input,
    int n) {
  __shared__ float smem[256];
  int tid = threadIdx.x;

  // Strided scan
  float local_max = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
    local_max = fmaxf(local_max, fabsf(input[i]));
  }

  smem[tid] = local_max;
  __syncthreads();

  // Block reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }

  if (tid == 0) {
    atomicMaxFloat(scale, smem[0] / FP8_E4M3_MAX);
  }
}

// Pass 2: quantize using the dynamically computed scale
__global__ void dynamic_fp8_quant(
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ scale,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float inv_scale = 1.0f / (*scale);
  float val = in[i] * inv_scale;
  val = fmaxf(-FP8_E4M3_MAX, fminf(val, FP8_E4M3_MAX));
  out[i] = __nv_fp8_e4m3(val);
}

__global__ void fp8_dequant(
    float* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ in,
    const float* __restrict__ scale,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = float(in[i]) * (*scale);
}

TEST(CUDA, QuantFP8Dynamic) {
  constexpr int n = 4096;
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  float h_in[n];
  for (int i = 0; i < n; i++) {
    h_in[i] = sinf(i * 0.01f) * 100.0f;  // dynamic range [-100, 100]
  }

  float *d_in, *d_out, *d_scale;
  __nv_fp8_e4m3* d_fp8;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMalloc(&d_fp8, n * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_scale, sizeof(float));

  cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_scale, 0, sizeof(float));  // must init to 0

  // Two-pass: find scale, then quantize
  find_absmax_and_scale<<<blocks, threads>>>(d_scale, d_in, n);
  dynamic_fp8_quant<<<blocks, threads>>>(d_fp8, d_in, d_scale, n);
  fp8_dequant<<<blocks, threads>>>(d_out, d_fp8, d_scale, n);
  cudaDeviceSynchronize();

  // Verify
  float h_out[n], h_scale;
  cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_GT(h_scale, 0.0f);

  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    max_err = fmaxf(max_err, fabsf(h_out[i] - h_in[i]));
  }
  EXPECT_LT(max_err, h_scale * FP8_E4M3_MAX * 0.02f);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_fp8);
  cudaFree(d_scale);
}
